# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch

# constants
DC = ScannetDatasetConfig()

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8] # put larger weights on positive objectness

def compute_vote_loss(data_dict):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        data_dict: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = data_dict["seed_xyz"].shape[0]
    num_seed = data_dict["seed_xyz"].shape[1] # B,num_seed,3
    vote_xyz = data_dict["vote_xyz"] # B,num_seed*vote_factor,3
    seed_inds = data_dict["seed_inds"].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(data_dict["vote_label_mask"], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(data_dict["vote_label"], 1, seed_inds_expand)
    seed_gt_votes += data_dict["seed_xyz"].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    
    return vote_loss

def compute_objectness_loss(data_dict):
    """ Compute objectness loss for the proposals.

    Args:
        data_dict: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # # Associate proposal and GT objects by point-to-point distances
    # aggregated_vote_xyz = data_dict["aggregated_vote_xyz"]
    # gt_center = data_dict["center_label"][:,:,0:3]
    # B = gt_center.shape[0]
    # K = aggregated_vote_xyz.shape[1]
    # K2 = gt_center.shape[1]
    # dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2
    # Associate proposal and GT objects by point-to-point distances
    pred_center = data_dict["center"]
    gt_center = data_dict["center_label"][:,:,0:3]
    B = gt_center.shape[0]
    K = pred_center.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    # NOTE don't consider gray zone for mask votenet
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    # objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    # objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = data_dict["objectness_scores"]
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction="none")
    # objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    # objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    objectness_loss = criterion(objectness_scores.view(B*K,-1), objectness_label.view(-1))
    objectness_loss = torch.mean(objectness_loss)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(data_dict, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        data_dict: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    # object_assignment = data_dict["object_assignment"]
    # batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = data_dict["center"]
    gt_center = data_dict["center_label"][:,:,0:3]
    batch_size = pred_center.shape[0]
    dist1, _, _, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    center_loss = torch.mean(torch.sqrt(dist1))

    # Compute size loss
    size_class_label = data_dict["size_class_label"] # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction="none")
    size_class_loss = criterion_size_class(data_dict["size_scores"].view(batch_size, -1), size_class_label.view(-1)) # (B,K)
    size_class_loss = torch.mean(size_class_loss)

    size_residual_label = data_dict["size_residual_label"] # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it"s *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(data_dict["size_residuals_normalized"]*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.mean(size_residual_normalized_loss)

    # 3.4 Semantic cls loss
    sem_cls_label = data_dict["sem_cls_label"] # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction="none")
    sem_cls_loss = criterion_sem_cls(data_dict["sem_cls_scores"].view(batch_size, -1), sem_cls_label.view(-1)) # (B,K)
    sem_cls_loss = torch.mean(sem_cls_loss)

    # Semantic cls acc
    sem_cls_pred = data_dict["sem_cls_scores"].argmax(-1) # (B, K)
    sem_cls_matched = (sem_cls_pred == sem_cls_label).view(-1).float()
    sem_cls_acc = torch.sum(sem_cls_matched)/(sem_cls_matched.shape[0]+1e-6)

    return center_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss, sem_cls_acc

def get_miou(data_dict):
    # predicted bbox
    pred_center = data_dict["center"].detach().cpu().numpy() # (B,K,3)
    pred_size_class = torch.argmax(data_dict["size_scores"], -1) # B,num_proposal
    pred_size_residual = torch.gather(data_dict["size_residuals"], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_class = pred_size_class.detach().cpu().numpy()
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    batch_size = pred_center.shape[0]

    # read GT bbox
    gt_bbox_batch = data_dict["bbox_corner"].cpu().numpy() # (B, 8, 3)

    # convert the bbox parameters to bbox corners
    pred_obb_batch = DC.param2obb_batch(pred_center[:, 0, 0:3], np.zeros((batch_size)), np.zeros((batch_size)),
                pred_size_class[:, 0], pred_size_residual[:, 0])
    pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
    
    ious = box3d_iou_batch(pred_bbox_batch, gt_bbox_batch)

    return np.mean(ious)

def get_loss(data_dict, config):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(data_dict)
    data_dict["vote_loss"] = vote_loss

    # Obj loss
    data_dict["objectness_loss"] = torch.zeros(1)[0].cuda()
    data_dict["objectness_label"] = torch.zeros(1)[0].cuda()
    data_dict["objectness_mask"] = torch.zeros(1)[0].cuda()
    data_dict["object_assignment"] = torch.zeros(1)[0].cuda()
    data_dict["pos_ratio"] = torch.zeros(1)[0].cuda()
    data_dict["neg_ratio"] = torch.zeros(1)[0].cuda()

    # Box loss and sem cls loss
    center_loss, size_cls_loss, size_reg_loss, sem_cls_loss, sem_cls_acc = compute_box_and_sem_cls_loss(data_dict, config)
    box_loss = center_loss + 0.1*size_cls_loss + size_reg_loss
    data_dict["center_loss"] = center_loss
    data_dict["heading_cls_loss"] = torch.zeros(1)[0].cuda()
    data_dict["heading_reg_loss"] = torch.zeros(1)[0].cuda()
    data_dict["size_cls_loss"] = size_cls_loss
    data_dict["size_reg_loss"] = size_reg_loss
    data_dict["sem_cls_loss"] = sem_cls_loss
    data_dict["box_loss"] = box_loss

    # objectness
    data_dict["objn_acc"] = torch.zeros(1)[0].cuda()

    # Final loss function
    loss = data_dict["vote_loss"] + 0.5*data_dict["objectness_loss"] + data_dict["box_loss"] + 0.1*data_dict["sem_cls_loss"]
    loss *= 10 # amplify

    # dump
    data_dict["sem_cls_acc"] = sem_cls_acc
    data_dict["miou"] = get_miou(data_dict)
    data_dict["loss"] = loss

    return data_dict
