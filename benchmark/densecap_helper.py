""" 
Helper functions and class to calculate Average Precisions for 3D object detection.

Modified from: https://github.com/facebookresearch/votenet/blob/master/models/ap_helper.py
"""

import os
import sys
import numpy as np

from benchmark.eval_densecap import eval_densecap, get_iou_obb

def flip_axis_to_camera(pc):
    """ Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    """
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[...,2] *= -1
    return pc2

def softmax(x):
    """ Numpy function for softmax"""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def parse_densecap_predictions(end_points):
    """ Parse predictions to OBB parameters and suppress overlapping boxes
    
    Args:
        end_points: dict
            {point_clouds, boxes, sem_prob, obj_prob, captions}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """

    pred_corners_3d_upright_camera = end_points["boxes"] # B,num_proposal,8,3
    pred_captions = end_points["captions"] # list of [caption]
    obj_prob = end_points["obj_prob"][:,:,1] # (B,K)

    bsize = pred_corners_3d_upright_camera.shape[0]

    batch_pred_map_cls = [] # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        batch_pred_map_cls.append([(pred_corners_3d_upright_camera[i,j], obj_prob[i,j], pred_captions[i][j]) \
            for j in range(pred_corners_3d_upright_camera.shape[1])])
    end_points["batch_pred_map_cls"] = batch_pred_map_cls

    return batch_pred_map_cls

def parse_densecap_groundtruths(end_points):
    """ Parse groundtruth labels to OBB parameters.
    
    Args:
        end_points: dict
            {box_label, caption_label}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """

    gt_corners_3d_upright_camera = end_points["box_label"]
    caption_labels = end_points["caption_label"]
    bsize = gt_corners_3d_upright_camera.shape[0]

    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append([(gt_corners_3d_upright_camera[i,j], caption_labels[i][j]) for j in range(gt_corners_3d_upright_camera.shape[1])])
    end_points["batch_gt_map_cls"] = batch_gt_map_cls

    return batch_gt_map_cls

class DenseCapAPCalculator(object):
    """ Calculating Average Precision """
    def __init__(self, iou_thresholds=[.1, .2, .3, .4, .5], meteor_threshold=[0, .05, .1, .15, .2, .25]):
        """
        Args:
            iou_thresholds: list of floats between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            meteor_threshold: list of floats between 0 and 1.0
                METEOR threshold to judge whether a prediction is positive.
        """

        self.iou_thresholds = iou_thresholds
        self.meteor_threshold = meteor_threshold
        self.build_thresholds(iou_thresholds, meteor_threshold)
        self.reset()
    
    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
        
        bsize = len(batch_pred_map_cls)
        assert(bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i] 
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i] 
            self.scan_cnt += 1
    
    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        
        recalls, precision, aps = [], [], []
        cache = None # no cache for the first time
        for thresholds in self.thresholds:
            rec, prec, ap, cache = eval_densecap(self.pred_map_cls, self.gt_map_cls, 
                thresholds=thresholds, get_iou_func=get_iou_obb, cache=cache)

            recalls.append(rec)
            precision.append(prec)
            aps.append(ap)
    
        ap_dict = {}
        for i in range(len(self.thresholds)):
            iou, meteor = self.thresholds[i]
            if iou not in ap_dict: ap_dict[iou] = {}
            if meteor not in ap_dict[iou]: ap_dict[iou][meteor] = 0

            ap_dict[iou][meteor] = aps[i]

        ret_dict = {
            "AP": ap_dict,
            "mAP": np.mean(aps)
        } 

        return ret_dict

    def build_thresholds(self, iou_thresholds, meteor_threshold):
        self.thresholds = []
        for iou in iou_thresholds:
            for meteor in meteor_threshold:
                self.thresholds.append((iou, meteor))

    def reset(self):
        self.gt_map_cls = {} # {scan_id: [(bbox, caption_list)]}
        self.pred_map_cls = {} # {scan_id: [(bbox, score, caption_list)]}
        self.scan_cnt = 0
