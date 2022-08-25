""" 
    Generic Code for Object Detection Evaluation
    From: https://github.com/facebookresearch/votenet/blob/master/utils/eval_det.py

    Input:
    For each class:
        For each image:
            Predictions: box, score
            Groundtruths: box
    
    Output:
    For each class:
        precision-recal and average precision
    
    Author: Charles R. Qi
    
    Ref: https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/datasets/voc_eval.py
"""

import os
import sys

import numpy as np

from multiprocessing import Pool
from nltk.translate.meteor_score import meteor_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from benchmark.box_util import box3d_iou
from benchmark.metric_util import calc_iou # axis-aligned 3D box IoU

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_iou(bb1, bb2):
    """ Compute IoU of two bounding boxes.
        ** Define your bod IoU function HERE **
    """
    #pass
    iou3d = calc_iou(bb1, bb2)
    return iou3d

def get_iou_obb(bb1,bb2):
    iou3d = box3d_iou(bb1, bb2)
    return iou3d

def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)

def eval_densecap(pred, gt, thresholds=(0.25, 0.25), use_07_metric=False, get_iou_func=get_iou, cache=None):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score, caption_list)]} where bbox is numpy array
            gt: map of {img_id: [bbox, caption_list]}
            thresholds: tuple of scalars, (iou threshold, METEOR threshold)
            use_07_metric: bool, if True use VOC07 11 point method
            cache: tuple of lists of IoU and METEOR between every predictions and GTs
                input None for the first calculation
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
    """

    # construct gt object and caption pairs
    class_recs = {} # {img_id: {"bbox": bbox list, "caption": caption list, "det": matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array([v[0] for v in gt[img_id]])
        caption = [v[1] for v in gt[img_id]]
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {"bbox": bbox, "caption": caption, "det": det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {"bbox": np.array([]), "caption": [], "det": []}

    # construct predictions
    image_ids = []
    confidence = [] # box confidence scores
    BB = [] # predicted bounding boxes
    CAP = [] # predicted captions
    for img_id in pred.keys():
        for box, score, caption_list in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
            CAP.append(caption_list)
    confidence = np.array(confidence)
    BB = np.array(BB) # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    CAP = [CAP[x] for x in sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]

    if cache is None: 
        iou_cache, meteor_cache = [], []
    else:
        iou_cache, meteor_cache = cache

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        #if d%100==0: print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d,...].astype(float)
        cap = CAP[d]
        ovmax = -np.inf
        nlpmax = -np.inf
        BBGT = R["bbox"].astype(float)
        CAPGT = R["caption"]

        if cache is None:
            cur_iou_cache = []
            cur_meteor_cache = []

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                if cache is None:
                    iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                    meteor = meteor_score(CAPGT[j], cap)
                    
                    cur_iou_cache.append(iou)
                    cur_meteor_cache.append(meteor)
                else:
                    iou = iou_cache[d][j]
                    meteor = meteor_cache[d][j]

                if iou > ovmax and meteor > nlpmax:
                    ovmax = iou
                    nlpmax = meteor
                    jmax = j
        
        # cache
        if cache is None:
            iou_cache.append(cur_iou_cache)
            meteor_cache.append(cur_meteor_cache)

        #print d, ovmax
        if ovmax > thresholds[0] and nlpmax > thresholds[1]:
            if not R["det"][jmax]:
                tp[d] = 1.
                R["det"][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos + 1e-8)
    #print("NPOS: ", npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    cache = (iou_cache, meteor_cache)

    return rec, prec, ap, cache

