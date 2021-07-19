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
from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch
from lib.config import CONF

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
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = data_dict["aggregated_vote_xyz"]
    gt_center = data_dict["center_label"][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = data_dict["objectness_scores"]
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction="none")
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

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

    object_assignment = data_dict["object_assignment"]
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = data_dict["center"]
    gt_center = data_dict["center_label"][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = data_dict["box_label_mask"]
    objectness_label = data_dict["objectness_label"].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(data_dict["heading_class_label"], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction="none")
    heading_class_loss = criterion_heading_class(data_dict["heading_scores"].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(data_dict["heading_residual_label"], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it"s *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(data_dict["heading_residuals_normalized"]*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(data_dict["size_class_label"], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction="none")
    size_class_loss = criterion_size_class(data_dict["size_scores"].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(data_dict["size_residual_label"], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it"s *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(data_dict["size_residuals_normalized"]*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(data_dict["sem_cls_label"], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction="none")
    sem_cls_loss = criterion_sem_cls(data_dict["sem_cls_scores"].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def compute_cap_loss(data_dict, config, weights):
    """ Compute cluster caption loss

    Args:
        data_dict: dict (read-only)

    Returns:
        cap_loss, cap_acc
    """

    # unpack
    pred_caps = data_dict["lang_cap"] # (B, num_words - 1, num_vocabs)
    num_words = data_dict["lang_len"].max()
    target_caps = data_dict["lang_ids"][:, 1:num_words] # (B, num_words - 1)
    
    _, _, num_vocabs = pred_caps.shape

    # caption loss
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    cap_loss = criterion(pred_caps.reshape(-1, num_vocabs), target_caps.reshape(-1))

    # mask out bad boxes
    good_bbox_masks = data_dict["good_bbox_masks"].unsqueeze(1).repeat(1, num_words-1) # (B, num_words - 1)
    good_bbox_masks = good_bbox_masks.reshape(-1) # (B * num_words - 1)
    cap_loss = torch.sum(cap_loss * good_bbox_masks) / (torch.sum(good_bbox_masks) + 1e-6)

    num_good_bbox = data_dict["good_bbox_masks"].sum()
    if num_good_bbox > 0: # only apply loss on the good boxes
        pred_caps = pred_caps[data_dict["good_bbox_masks"]] # num_good_bbox
        target_caps = target_caps[data_dict["good_bbox_masks"]] # num_good_bbox

        # caption acc
        pred_caps = pred_caps.reshape(-1, num_vocabs).argmax(-1) # num_good_bbox * (num_words - 1)
        target_caps = target_caps.reshape(-1) # num_good_bbox * (num_words - 1)
        masks = target_caps != 0
        masked_pred_caps = pred_caps[masks]
        masked_target_caps = target_caps[masks]
        cap_acc = (masked_pred_caps == masked_target_caps).sum().float() / masks.sum().float()
    else: # zero placeholder if there is no good box
        cap_acc = torch.zeros(1)[0].cuda()
    
    return cap_loss, cap_acc

def radian_to_label(radians, num_bins=6):
    """
        convert radians to labels

        Arguments:
            radians: a tensor representing the rotation radians, (batch_size)
            radians: a binary tensor representing the valid masks, (batch_size)
            num_bins: number of bins for discretizing the rotation degrees

        Return:
            labels: a long tensor representing the discretized rotation degree classes, (batch_size)
    """

    boundaries = torch.arange(np.pi / num_bins, np.pi-1e-8, np.pi / num_bins).cuda()
    labels = torch.bucketize(radians, boundaries)

    return labels

def compute_node_orientation_loss(data_dict, num_bins=6):
    object_assignment = data_dict["object_assignment"]
    edge_indices = data_dict["edge_index"]
    edge_preds = data_dict["edge_orientations"]
    num_sources = data_dict["num_edge_source"]
    num_targets = data_dict["num_edge_target"]
    batch_size, num_proposals = object_assignment.shape

    object_rotation_matrices = torch.gather(
        data_dict["scene_object_rotations"], 
        1, 
        object_assignment.view(batch_size, num_proposals, 1, 1).repeat(1, 1, 3, 3)
    ) # batch_size, num_proposals, 3, 3
    object_rotation_masks = torch.gather(
        data_dict["scene_object_rotation_masks"], 
        1, 
        object_assignment
    ) # batch_size, num_proposals
    
    preds = []
    labels = []
    masks = []
    for batch_id in range(batch_size):
        batch_rotations = object_rotation_matrices[batch_id] # num_proposals, 3, 3
        batch_rotation_masks = object_rotation_masks[batch_id] # num_proposals

        batch_num_sources = num_sources[batch_id]
        batch_num_targets = num_targets[batch_id]
        batch_edge_indices = edge_indices[batch_id, :batch_num_sources * batch_num_targets]

        source_indices = edge_indices[batch_id, 0, :batch_num_sources*batch_num_targets].long()
        target_indices = edge_indices[batch_id, 1, :batch_num_sources*batch_num_targets].long()

        source_rot = torch.index_select(batch_rotations, 0, source_indices)
        target_rot = torch.index_select(batch_rotations, 0, target_indices)

        relative_rot = torch.matmul(source_rot, target_rot.transpose(2, 1))
        relative_rot = torch.acos(torch.clamp(0.5 * (torch.diagonal(relative_rot, dim1=-2, dim2=-1).sum(-1) - 1), -1, 1))
        assert torch.isfinite(relative_rot).sum() == source_indices.shape[0]

        source_masks = torch.index_select(batch_rotation_masks, 0, source_indices)
        target_masks = torch.index_select(batch_rotation_masks, 0, target_indices)
        batch_edge_masks = source_masks * target_masks
        
        batch_edge_labels = radian_to_label(relative_rot, num_bins)
        batch_edge_preds = edge_preds[batch_id, :batch_num_sources * batch_num_targets]

        preds.append(batch_edge_preds)
        labels.append(batch_edge_labels)
        masks.append(batch_edge_masks)

    # aggregate
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    masks = torch.cat(masks, dim=0)

    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(preds, labels)
    loss = (loss * masks).sum() / (masks.sum() + 1e-8)

    preds = preds.argmax(-1)
    acc = (preds[masks==1] == labels[masks==1]).sum().float() / (masks.sum().float() + 1e-8)

    return loss, acc

def compute_node_distance_loss(data_dict):
    gt_center = data_dict["center_label"][:,:,0:3]
    object_assignment = data_dict["object_assignment"]
    
    gt_center = torch.gather(gt_center, 1, object_assignment.unsqueeze(-1).repeat(1, 1, 3))
    batch_size, _, _ = gt_center.shape

    edge_indices = data_dict["edge_index"]
    edge_preds = data_dict["edge_distances"]
    num_sources = data_dict["num_edge_source"]
    num_targets = data_dict["num_edge_target"]

    preds = []
    labels = []
    for batch_id in range(batch_size):
        batch_gt_center = gt_center[batch_id]

        batch_num_sources = num_sources[batch_id]
        batch_num_targets = num_targets[batch_id]
        batch_edge_indices = edge_indices[batch_id, :batch_num_sources * batch_num_targets]

        source_indices = edge_indices[batch_id, 0, :batch_num_sources*batch_num_targets].long()
        target_indices = edge_indices[batch_id, 1, :batch_num_sources*batch_num_targets].long()

        source_centers = torch.index_select(batch_gt_center, 0, source_indices)
        target_centers = torch.index_select(batch_gt_center, 0, target_indices)

        batch_edge_labels = torch.norm(source_centers - target_centers, dim=1)
        batch_edge_preds = edge_preds[batch_id, :batch_num_sources * batch_num_targets]

        preds.append(batch_edge_preds)
        labels.append(batch_edge_labels)

    # aggregate
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    criterion = nn.MSELoss()
    loss = criterion(preds, labels)

    return loss

def compute_object_cls_loss(data_dict, weights):
    """ Compute object classification loss

    Args:
        data_dict: dict (read-only)

    Returns:
        cls_loss, cls_acc
    """

    # unpack
    preds = data_dict["enc_preds"] # (B, num_cls)
    targets = data_dict["object_cat"] # (B,)
    
    # classification loss
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).cuda())
    cls_loss = criterion(preds, targets)

    # classification acc
    preds = preds.argmax(-1) # (B,)
    cls_acc = (preds == targets).sum().float() / targets.shape[0]

    return cls_loss, cls_acc

def get_scene_cap_loss(data_dict, device, config, weights, 
    detection=True, caption=True, orientation=False, distance=False, num_bins=CONF.TRAIN.NUM_BINS):
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

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    num_proposal = objectness_label.shape[1]
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    data_dict["objectness_label"] = objectness_label
    data_dict["objectness_mask"] = objectness_mask
    data_dict["object_assignment"] = object_assignment
    data_dict["pos_ratio"] = torch.sum(objectness_label.float().to(device))/float(total_num_proposal)
    data_dict["neg_ratio"] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict["pos_ratio"]

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss

    # objectness
    obj_pred_val = torch.argmax(data_dict["objectness_scores"], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==data_dict["objectness_label"].long()).float()*data_dict["objectness_mask"])/(torch.sum(data_dict["objectness_mask"])+1e-6)
    data_dict["obj_acc"] = obj_acc

    if detection:
        data_dict["vote_loss"] = vote_loss
        data_dict["objectness_loss"] = objectness_loss
        data_dict["center_loss"] = center_loss
        data_dict["heading_cls_loss"] = heading_cls_loss
        data_dict["heading_reg_loss"] = heading_reg_loss
        data_dict["size_cls_loss"] = size_cls_loss
        data_dict["size_reg_loss"] = size_reg_loss
        data_dict["sem_cls_loss"] = sem_cls_loss
        data_dict["box_loss"] = box_loss
    else:
        data_dict["vote_loss"] = torch.zeros(1)[0].to(device)
        data_dict["objectness_loss"] = torch.zeros(1)[0].to(device)
        data_dict["center_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_reg_loss"] = torch.zeros(1)[0].to(device)
        data_dict["size_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["size_reg_loss"] = torch.zeros(1)[0].to(device)
        data_dict["sem_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["box_loss"] = torch.zeros(1)[0].to(device)

    if caption:
        cap_loss, cap_acc = compute_cap_loss(data_dict, config, weights)

        # store
        data_dict["cap_loss"] = cap_loss
        data_dict["cap_acc"] = cap_acc
    else:
        # store
        data_dict["cap_loss"] = torch.zeros(1)[0].to(device)
        data_dict["cap_acc"] = torch.zeros(1)[0].to(device)
        data_dict["pred_ious"] =  torch.zeros(1)[0].to(device)

    if orientation:
        ori_loss, ori_acc = compute_node_orientation_loss(data_dict, num_bins)

        # store
        data_dict["ori_loss"] = ori_loss
        data_dict["ori_acc"] = ori_acc
    else:
        # store
        data_dict["ori_loss"] = torch.zeros(1)[0].to(device)
        data_dict["ori_acc"] = torch.zeros(1)[0].to(device)

    if distance:
        dist_loss = compute_node_distance_loss(data_dict)

        # store
        data_dict["dist_loss"] = dist_loss
    else:
        # store
        data_dict["dist_loss"] = torch.zeros(1)[0].to(device)

    # Final loss function
    # loss = data_dict["vote_loss"] + 0.5*data_dict["objectness_loss"] + data_dict["box_loss"] + 0.1*data_dict["sem_cls_loss"] + data_dict["cap_loss"]

    if detection:
        loss = data_dict["vote_loss"] + 0.5*data_dict["objectness_loss"] + data_dict["box_loss"] + 0.1*data_dict["sem_cls_loss"]
        loss *= 10 # amplify
        if caption:
            loss += data_dict["cap_loss"]
        if orientation:
            loss += 0.1*data_dict["ori_loss"]
        if distance:
            loss += 0.1*data_dict["dist_loss"]
            # loss += data_dict["dist_loss"]
    else:
        loss = data_dict["cap_loss"]
        if orientation:
            loss += 0.1*data_dict["ori_loss"]
        if distance:
            loss += 0.1*data_dict["dist_loss"]

    data_dict["loss"] = loss

    return data_dict

def get_object_cap_loss(data_dict, config, weights, classify=True, caption=True):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    if classify:
        cls_loss, cls_acc = compute_object_cls_loss(data_dict, weights)

        data_dict["cls_loss"] = cls_loss
        data_dict["cls_acc"] = cls_acc
    else:
        data_dict["cls_loss"] = torch.zeros(1)[0].cuda()
        data_dict["cls_acc"] = torch.zeros(1)[0].cuda()

    if caption:
        cap_loss, cap_acc = compute_cap_loss(data_dict, config, weights)

        # store
        data_dict["cap_loss"] = cap_loss
        data_dict["cap_acc"] = cap_acc
    else:
        # store
        data_dict["cap_loss"] = torch.zeros(1)[0].cuda()
        data_dict["cap_acc"] = torch.zeros(1)[0].cuda()

    # Final loss function
    loss = data_dict["cls_loss"] + data_dict["cap_loss"]
    
    # loss *= 10 # amplify

    data_dict["loss"] = loss

    return data_dict
