import os
import sys
import json
import torch
import argparse

import numpy as np

from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import benchmark.capeval.bleu.bleu as capblue
import benchmark.capeval.cider.cider as capcider
import benchmark.capeval.rouge.rouge as caprouge
import benchmark.capeval.meteor.meteor as capmeteor

from benchmark.box_util import box3d_iou_batch_tensor, generalized_box3d_iou
from benchmark.scannet_utils import ScannetDatasetConfig
from benchmark.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from benchmark.densecap_helper import DenseCapAPCalculator, parse_densecap_predictions, parse_densecap_groundtruths

SCANREFER_GT = "/cluster/balrog/dchen/ScanRefer/data/ScanRefer_filtered_{}_gt_bbox.json" # TODO change this; split

NYU40IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40] # exclude wall (1), floor (2), ceiling (22)
TYPE2LABEL = {
    'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
    'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
    'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17
}
SCANNET_LABEL_MAP = "./data/scannet/meta_data/scannetv2-labels.combined.tsv" # TODO change this

# constants
DC = ScannetDatasetConfig()

# results template
CAP_TEMPLATE = """
| CIDEr@0.25IoU | BLEU-4@0.25IoU | ROUGE-L@0.25IoU | METEOR@0.25IoU |
|    {:.4f}     |     {:.4f}     |     {:.4f}      |     {:.4f}     |

| CIDEr@0.5IoU  | BLEU-4@0.5IoU  | ROUGE-L@0.5IoU  | METEOR@0.5IoU  |
|    {:.4f}     |     {:.4f}     |     {:.4f}      |     {:.4f}     |
"""

DET_TEMPLATE = """
| mAP@0.25 | mAP@0.5  |
|  {:.4f}  |  {:.4f}  |
"""

def prepare_corpus(gts):
    corpus = {}
    for scene_id, value in gts.items():
        for gt_id, object_id in enumerate(value["object_ids"]):
            key = "{}|{}".format(scene_id, object_id)
            captions = value["captions"][gt_id]

            corpus[key] = captions

    return corpus

def filter_candidates(candidates, min_iou):
    # new_candidates = {}
    # for key, value in candidates.items():
    #     if value["iou"] >= min_iou:
    #         new_candidates[key] = [value["caption"]]

    # return new_candidates

    masks = []
    new_candidates = {}
    for key, value in candidates.items():
        if value["iou"] >= min_iou:
            masks.append(1)
        else:
            masks.append(0)

        new_candidates[key] = [value["caption"]]

    return np.array(masks), new_candidates

def check_candidates(corpus, candidates, special_tokens):
    placeholder = "{} {}".format(special_tokens["bos_token"], special_tokens["eos_token"])
    corpus_keys = list(corpus.keys())
    candidate_keys = list(candidates.keys())
    missing_keys = [key for key in corpus_keys if key not in candidate_keys]

    if len(missing_keys) != 0:
        for key in missing_keys:
            candidates[key] = [placeholder]

    return candidates

def organize_candidates(corpus, candidates):
    new_candidates = {}
    for key in corpus.keys():
        new_candidates[key] = candidates[key]

    return new_candidates

def organize_prediction(predictions):
    organized = {} # "scene_id" -> {"captions" -> [], "boxes" -> array}
    for scene_id, data in predictions.items():
        captions, boxes, sem_prob, obj_prob = [], [], [], []
        for pred in data:
            captions.append(pred["caption"])
            boxes.append(pred["box"])
            sem_prob.append(pred["sem_prob"])
            obj_prob.append(pred["obj_prob"])

        organized[scene_id] = {
            "captions": captions, # should be a list of length M
            "boxes": np.array(boxes), # should be of shape (M, 8, 3)
            "sem_prob": np.array(sem_prob), # should be of shape (M, num_cls)
            "obj_prob": np.array(obj_prob) # should be of shape (M, 2)
        }

    return organized

def organize_gt(gts):
    # name2label
    lines = [line.rstrip() for line in open(SCANNET_LABEL_MAP)]
    lines = lines[1:]
    name2label = {}
    for i in range(len(lines)):
        label_classes_set = set(TYPE2LABEL.keys())
        elements = lines[i].split('\t')
        raw_name = elements[1]
        nyu40_id = int(elements[4])
        nyu40_name = elements[7]
        if nyu40_id in NYU40IDS:
            if nyu40_name not in label_classes_set:
                name2label[raw_name] = TYPE2LABEL["others"]
            else:
                name2label[raw_name] = TYPE2LABEL[nyu40_name]

    organized = {}
    for data in gts:
        scene_id = data["scene_id"]
        object_id = data["object_id"]

        if scene_id not in organized:
            organized[scene_id] = {}

        if object_id not in organized[scene_id]:
            organized[scene_id][object_id] = []

        organized[scene_id][object_id].append(data)

    new = {} # "scene_id" -> {"captions" -> [], "boxes" -> array}
    for scene_id in organized:
        captions, boxes, object_ids, object_sems = [], [], [], []
        for object_id, data in organized[scene_id].items():
            boxes.append(data[0]["bbox"]) # boxes are duplicated
            object_ids.append(data[0]["object_id"]) # object IDs are duplicated

            object_name = " ".join(data[0]["object_name"].split("_"))

            try:
                object_sems.append(name2label[object_name]) # object IDs are duplicated
            except KeyError:
                object_sems.append(17) # others

            object_captions = []
            for d in data:
                caption = " ".join(d["token"])
                object_captions.append("sos {} eos".format(caption))

            captions.append(object_captions)
        
        new[scene_id] = {
            "captions": captions, # should be a list of length M
            "boxes": np.array(boxes), # should be of shape (M, 8, 3)
            "object_ids": np.array(object_ids), # should be of shape (M,)
            "object_sems": np.array(object_sems) # should be of shape (M,)
        }

    return new

def box_assignment(pred_boxes, gt_boxes):
    nprop = torch.tensor([pred_boxes.shape[1]]).type_as(pred_boxes).long()
    nprop = nprop.unsqueeze(0)
    
    ngt = torch.tensor([gt_boxes.shape[1]]).type_as(pred_boxes).long()
    ngt = ngt.unsqueeze(0)

    gious = generalized_box3d_iou(
        pred_boxes,
        gt_boxes,
        ngt,
        rotated_boxes=False,
        needs_grad=False,
    ) # B, K1, K2

    # hungarian assignment
    final_cost = -gious.detach().cpu().numpy()

    assignments = []

    # assignments from proposals to GTs
    per_prop_gt_inds = torch.zeros(
        [1, nprop], dtype=torch.int64, device=pred_boxes.device
    )
    prop_matched_mask = torch.zeros(
        [1, nprop], dtype=torch.float32, device=pred_boxes.device
    )

    # assignments from GTs to proposals
    per_gt_prop_inds = torch.zeros(
        [1, ngt], dtype=torch.int64, device=pred_boxes.device
    )
    gt_matched_mask = torch.zeros(
        [1, ngt], dtype=torch.float32, device=pred_boxes.device
    )

    for b in range(1):
        assign = []
        assign = linear_sum_assignment(final_cost[b, :, : ngt[b]])
        assign = [
            torch.from_numpy(x).long().to(device=pred_boxes.device)
            for x in assign
        ]

        per_prop_gt_inds[b, assign[0]] = assign[1]
        prop_matched_mask[b, assign[0]] = 1

        per_gt_prop_inds[b, assign[1]] = assign[0]
        gt_matched_mask[b, assign[1]] = 1

        assignments.append(assign)

    return {
        "assignments": assignments,
        "per_prop_gt_inds": per_prop_gt_inds,
        "prop_matched_mask": prop_matched_mask,
        "per_gt_prop_inds": per_gt_prop_inds,
        "gt_matched_mask": gt_matched_mask
    }

def assign_pred_to_gt(predictions, gts):
    candidates = {}
    total_num_preds, total_num_gts = 0, 0
    for scene_id in gts:
        try:
            scene_preds = predictions[scene_id]
            scene_gts = gts[scene_id]

            pred_boxes = torch.tensor(scene_preds["boxes"]).unsqueeze(0) # 1, K1, 8, 3
            gt_boxes = torch.tensor(scene_gts["boxes"]).unsqueeze(0) # 1, K2, 8, 3
            batch_size, num_gts, *_ = gt_boxes.shape

            total_num_preds += pred_boxes.shape[1]
            total_num_gts += gt_boxes.shape[1]
            
            assignments = box_assignment(pred_boxes, gt_boxes)

            per_gt_prop_inds = assignments["per_gt_prop_inds"]
            matched_prop_box_corners = torch.gather(
                pred_boxes, 1, per_gt_prop_inds[:, :, None, None].repeat(1, 1, 8, 3)
            ) # 1, num_gts, 8, 3 
            matched_ious = box3d_iou_batch_tensor(
                matched_prop_box_corners.reshape(-1, 8, 3), 
                gt_boxes.reshape(-1, 8, 3)
            ).reshape(batch_size, num_gts)
            
            for gt_id in range(num_gts):
                caption = scene_preds["captions"][per_gt_prop_inds[0, gt_id]]
                object_id = scene_gts["object_ids"][gt_id]
                iou = matched_ious[0, gt_id].item()
                box = matched_prop_box_corners[0, gt_id].detach().cpu().numpy().tolist()
                gt_box = gt_boxes[0, gt_id].detach().cpu().numpy().tolist()

                key = "{}|{}".format(scene_id, str(object_id))
                entry = {
                    "caption": caption,
                    "iou": iou,
                    "box": box,
                    "gt_box": gt_box
                }

                if key not in candidates:
                    candidates[key] = entry
                else:
                    # update the stored prediction if IoU is higher
                    if iou > candidates[key][1]:
                        candidates[key] = entry
        
        except KeyError:
            pass

    return candidates, total_num_preds, total_num_gts

def aggregate_score(score_arr, masks, total_num):
    aggr_score = np.sum(score_arr * masks) / total_num

    return aggr_score

def compute_f1_score(precision, recall):
    f1_score = 2 * precision * recall
    f1_score /= (precision + recall)

    return f1_score

def evaluate_captioning(args, predictions, gts, min_ious=[0, 0.25, 0.5]):
    assigned_candidates, total_num_preds, total_num_gts = assign_pred_to_gt(predictions, gts)

    corpus = prepare_corpus(gts)

    results = {}
    for min_iou in min_ious:
        # check candidates
        # NOTE: make up the captions for the undetected object by "sos eos"
        masks, candidates = filter_candidates(assigned_candidates, min_iou)
        candidates = check_candidates(corpus, candidates, {"bos_token": "sos", "eos_token": "eos"})
        candidates = organize_candidates(corpus, candidates)

        # compute scores
        bleu = capblue.Bleu(4).compute_score(corpus, candidates)
        cider = capcider.Cider().compute_score(corpus, candidates)
        rouge = caprouge.Rouge().compute_score(corpus, candidates) 
        meteor = capmeteor.Meteor().compute_score(corpus, candidates)

        # aggregate recall
        precision_bleu = [
            aggregate_score(bleu[1][0], masks, total_num_preds), 
            aggregate_score(bleu[1][1], masks, total_num_preds), 
            aggregate_score(bleu[1][2], masks, total_num_preds), 
            aggregate_score(bleu[1][3], masks, total_num_preds)
        ]
        precision_cider = aggregate_score(cider[1], masks, total_num_preds)
        precision_rouge = aggregate_score(rouge[1], masks, total_num_preds)
        precision_meteor = aggregate_score(meteor[1], masks, total_num_preds)

        # aggregate recall
        recall_bleu = [
            aggregate_score(bleu[1][0], masks, total_num_gts), 
            aggregate_score(bleu[1][1], masks, total_num_gts), 
            aggregate_score(bleu[1][2], masks, total_num_gts), 
            aggregate_score(bleu[1][3], masks, total_num_gts)
        ]
        recall_cider = aggregate_score(cider[1], masks, total_num_gts)
        recall_rouge = aggregate_score(rouge[1], masks, total_num_gts)
        recall_meteor = aggregate_score(meteor[1], masks, total_num_gts)

        if args.verbose:
            # report
            print("\n----------------------Evaluation @ {} IoU-----------------------".format(min_iou))
            print("[BLEU-1] Precision: {:.4f},  Recall: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(precision_bleu[0], recall_bleu[0], max(bleu[1][0]), min(bleu[1][0])))
            print("[BLEU-2] Precision: {:.4f},  Recall: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(precision_bleu[1], recall_bleu[1], max(bleu[1][1]), min(bleu[1][1])))
            print("[BLEU-3] Precision: {:.4f},  Recall: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(precision_bleu[2], recall_bleu[2], max(bleu[1][2]), min(bleu[1][2])))
            print("[BLEU-4] Precision: {:.4f},  Recall: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(precision_bleu[3], recall_bleu[3], max(bleu[1][3]), min(bleu[1][3])))
            print("[CIDEr] Precision: {:.4f},  Recall: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(precision_cider, recall_cider, max(cider[1]), min(cider[1])))
            print("[ROUGE-L] Precision: {:.4f},  Recall: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(precision_rouge, recall_rouge, max(rouge[1]), min(rouge[1])))
            print("[METEOR] Precision: {:.4f},  Recall: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(precision_meteor, recall_meteor, max(meteor[1]), min(meteor[1])))
            print()

        results[min_iou] = {
            "precision": {
                "bleu": precision_bleu,
                "cider": precision_cider,
                "rouge": precision_rouge,
                "meteor": precision_meteor,
            },
            "recall": {
                "bleu": recall_bleu,
                "cider": recall_cider,
                "rouge": recall_rouge,
                "meteor": recall_meteor,
            },
            "f1-score": {
                "bleu": [compute_f1_score(x, y) for x, y in zip(precision_bleu, recall_bleu)],
                "cider": compute_f1_score(precision_cider, recall_cider),
                "rouge": compute_f1_score(precision_rouge, recall_rouge),
                "meteor": compute_f1_score(precision_meteor, recall_meteor),
            }
            
        }

    return results

def evaluate_dense_captioning(args, predictions, gts):
    IOU_THRESHOLDS = [.1, .2, .3, .4, .5]
    METEOR_THRESHOLDS = [.15, .3, .45, .6, .75]
    AP_CALCULATOR = DenseCapAPCalculator(IOU_THRESHOLDS, METEOR_THRESHOLDS)

    for scene_id in gts.keys():
        end_preds = {
            "boxes": predictions[scene_id]["boxes"][None, ...], # should be of shape (1, M, 8, 3)
            "obj_prob": predictions[scene_id]["obj_prob"][None, ...], # should be of shape (1, M, 2)
            "captions": [predictions[scene_id]["captions"]] # should be of length 1
        }
        end_gts = {
            "box_label": gts[scene_id]["boxes"][None, ...], # should be of shape (1, N, 8, 3)
            "caption_label": [gts[scene_id]["captions"]] # should be of length 1
        }

        batch_pred_map_cls = parse_densecap_predictions(end_preds) 
        batch_gt_map_cls = parse_densecap_groundtruths(end_gts) 
        AP_CALCULATOR.step(batch_pred_map_cls, batch_gt_map_cls)

    # aggregate dense captioning results and report
    results = AP_CALCULATOR.compute_metrics()

    if args.verbose:
        iou_list = list(results["AP"].keys())
        meteor_list = list(results["AP"][iou_list[0]].keys())

        # head
        print()
        print("              ", end="|")
        for meteor in meteor_list:
            print(" METEOR: {:.4f} ".format(meteor), end="|")
        print()

        # body
        for iou in iou_list:
            print("| IoU: {:.4f} ".format(iou), end="|")
            for meteor in meteor_list:
                print("         {:.4f} ".format(results["AP"][iou][meteor]), end="|")
            print()

    return results

def evaluate_detection(args, predictions, gts):
    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    }
    AP_IOU_THRESHOLDS = [0.25, 0.5]
    AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    # print(sorted(list(predictions.keys())))
    # print(sorted(list(gts.keys())))
    # assert sorted(list(predictions.keys())) == sorted(list(gts.keys()))
    for scene_id in gts.keys():
        pred_boxes = {
            "boxes": predictions[scene_id]["boxes"][None, ...], # should be of shape (1, M, 8, 3)
            "sem_prob": predictions[scene_id]["sem_prob"][None, ...], # should be of shape (1, M, num_cls)
            "obj_prob": predictions[scene_id]["obj_prob"][None, ...] # should be of shape (1, M, 2)
        }
        gt_boxes = {
            "box_label": gts[scene_id]["boxes"][None, ...], # should be of shape (1, N, 8, 3)
            "sem_cls_label": gts[scene_id]["object_sems"][None, ...], # should be of shape (1, N)
        }

        batch_pred_map_cls = parse_predictions(pred_boxes, POST_DICT) 
        batch_gt_map_cls = parse_groundtruths(gt_boxes, POST_DICT) 
        for ap_calculator in AP_CALCULATOR_LIST:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # aggregate object detection results and report
    results = {}
    for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
        metrics_dict = ap_calculator.compute_metrics()
        results[AP_IOU_THRESHOLDS[i]] = metrics_dict

        if args.verbose:
            print()
            print("-"*10, "iou_thresh: %f"%(AP_IOU_THRESHOLDS[i]), "-"*10)
            for key in metrics_dict:
                print("eval %s: %f"%(key, metrics_dict[key]))
            print()

    return results

def evaluate(args, predictions, gts, min_ious=[0, 0.25, 0.5]):
    print("=> evaluating captioning...")
    cap_results = evaluate_captioning(args, predictions, gts, min_ious)

    print("=> evaluating object detection...")
    det_results = evaluate_detection(args, predictions, gts)

    print("=> evaluating dense captioning...")
    densecap_results = evaluate_dense_captioning(args, predictions, gts)

    # report
    for key in ["precision", "recall", "f1-score"]:
        print("\n==> Captioning {}:".format(key))
        print(CAP_TEMPLATE.format(
            # 0.25
            cap_results[0.25][key]["cider"], 
            cap_results[0.25][key]["bleu"][3], 
            cap_results[0.25][key]["rouge"], 
            cap_results[0.25][key]["meteor"],
            # 0.5
            cap_results[0.5][key]["cider"], 
            cap_results[0.5][key]["bleu"][3], 
            cap_results[0.5][key]["rouge"], 
            cap_results[0.5][key]["meteor"],
        ))

    print("\n==> Object Detection")
    print(DET_TEMPLATE.format(det_results[0.25]["mAP"], det_results[0.5]["mAP"]))

    print("\n==> Dense Captioning mAP: {:.4f}".format(densecap_results["mAP"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, help="Split of ScanRefer", default="test")
    parser.add_argument("--path", type=str, help="Path to the prediction json file", required=True)
    parser.add_argument("--verbose", action="store_true", help="Report all metrics.")
    args = parser.parse_args()

    with open(args.path) as f:
        raw_preds = json.load(f)
        predictions = organize_prediction(raw_preds)

    with open(SCANREFER_GT.format(args.split)) as f:
        raw_gts = json.load(f)
        gts = organize_gt(raw_gts)

    evaluate(args, predictions, gts)
