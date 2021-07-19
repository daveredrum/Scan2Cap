import os
import sys
import json
import torch
import pickle
import argparse

import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.config import CONF
from lib.ap_helper import parse_predictions
from lib.loss_helper_pretrained import get_loss
from utils.box_util import box3d_iou_batch_tensor
from utils.nn_distance import nn_distance

# constants
DC = ScannetDatasetConfig()

SCANREFER = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered.json")))
SCANREFER_ORGANIZED = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_organized.json")))

def prepare_corpus(scanrefer, max_len=CONF.TRAIN.MAX_DES_LEN):
    scene_ids = list(set([data["scene_id"] for data in scanrefer]))

    corpus = {}
    for data in SCANREFER:
        scene_id = data["scene_id"]

        if scene_id not in scene_ids: continue

        object_id = data["object_id"]
        object_name = data["object_name"]
        token = data["token"][:max_len]
        description = " ".join(token)

        # add start and end token
        description = "sos " + description
        description += " eos"

        key = "{}|{}|{}".format(scene_id, object_id, object_name)
        # key = "{}|{}".format(scene_id, object_id)

        if key not in corpus:
            corpus[key] = []

        corpus[key].append(description)

    return corpus

def decode_caption(raw_caption, idx2word):
    decoded = ["sos"]
    for token_idx in raw_caption:
        token_idx = token_idx.item()
        token = idx2word[str(token_idx)]
        decoded.append(token)
        if token == "eos": break

    if "eos" not in decoded: decoded.append("eos")
    decoded = " ".join(decoded)

    return decoded

def check_candidates(corpus, candidates):
    placeholder = "sos eos"
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

def filter_stop_words(corpus, candidates):
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')

    for key in corpus:
        corpus[key] = [w for w in corpus[key] if w not in stopwords.words()]
        candidates[key] = [w for w in candidates[key] if w not in stopwords.words()]

    return corpus, candidates

def feed_gt_cap(model, dataset, dataloader, phase, folder, use_tf=False, max_len=CONF.TRAIN.MAX_DES_LEN, save_interm=False):
    candidates = {}
    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        with torch.no_grad():
            data_dict = model(data_dict, use_tf, is_eval=True)

        # unpack
        captions = data_dict["lang_cap"].argmax(-1) # batch_size, max_len - 1
        dataset_ids = data_dict["dataset_idx"]
        object_ids = data_dict["bbox_object_ids"]
        bbox_mask = data_dict["bbox_mask"]
        # batch_size, _ = captions.shape
        batch_size, num_proposals, _ = captions.shape

        # dump generated captions
        # for batch_id in range(batch_size):
        #     dataset_idx = dataset_ids[batch_id].item()
        #     scene_id = dataset.scanrefer[dataset_idx]["scene_id"]
        #     object_id = dataset.scanrefer[dataset_idx]["object_id"]
        #     caption_decoded = decode_caption(captions[batch_id], dataset.vocabulary["idx2word"])

        #     try:
        #         ann_list = list(SCANREFER_ORGANIZED[scene_id][object_id].keys())
        #         object_name = SCANREFER_ORGANIZED[scene_id][object_id][ann_list[0]]["object_name"]

        #         # store
        #         key = "{}|{}|{}".format(scene_id, object_id, object_name)
        #         candidates[key] = [caption_decoded]
        #     except KeyError:
        #         continue

        for batch_id in range(batch_size):
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = dataset.scanrefer[dataset_idx]["scene_id"]
            for prop_id in range(num_proposals):
                if bbox_mask[batch_id, prop_id] == 1:
                    object_id = str(object_ids[batch_id, prop_id].item())
                    caption_decoded = decode_caption(captions[batch_id, prop_id], dataset.vocabulary["idx2word"])

                    # print(scene_id, object_id)
                    try:
                        ann_list = list(SCANREFER_ORGANIZED[scene_id][object_id].keys())
                        object_name = SCANREFER_ORGANIZED[scene_id][object_id][ann_list[0]]["object_name"]

                        # store
                        key = "{}|{}|{}".format(scene_id, object_id, object_name)
                        # key = "{}|{}".format(scene_id, object_id)
                        candidates[key] = [caption_decoded]

                    except KeyError:
                        continue

    return candidates

def feed_votenet_cap(model, dataset, dataloader, phase, folder, 
    use_tf=False, max_len=CONF.TRAIN.MAX_DES_LEN, save_interm=False, min_iou=CONF.EVAL.MIN_IOU_THRESHOLD):
    candidates = {}
    intermediates = {}
    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        with torch.no_grad():
            data_dict = model(data_dict, use_tf, True)

        # unpack
        captions = data_dict["lang_cap"].argmax(-1) # batch_size, num_proposals, max_len - 1
        dataset_ids = data_dict["dataset_idx"]
        batch_size, _, _ = captions.shape

        # object mask
        objn_masks = data_dict["bbox_mask"]

        # assign bboxes
        target_bbox_centers = data_dict["ref_box_center_label"] # batch_size, num_gt_boxes, 3
        detected_bbox_centers = data_dict["bbox_center"] # batch_size, num_proposals, 3
        _, num_gt_boxes, _, = target_bbox_centers.shape
        _, num_proposals, _, = detected_bbox_centers.shape
        _, assignment, _, _ = nn_distance(detected_bbox_centers, target_bbox_centers) # batch_size, num_proposals
        detected_object_ids = torch.gather(data_dict["scene_object_ids"], 1, assignment) # batch_size, num_proposals

        # bbox corners
        assigned_target_bbox_corners = torch.gather(
            data_dict["ref_box_corner_label"], 
            1, 
            assignment.view(batch_size, num_proposals, 1, 1).repeat(1, 1, 8, 3)
        ) # batch_size, num_proposals, 8, 3
        detected_bbox_corners = data_dict["bbox_corner"] # batch_size, num_proposals, 8, 3
        
        # compute IoU between each detected box and each ground truth box
        ious = box3d_iou_batch_tensor(
            assigned_target_bbox_corners.view(-1, 8, 3), # batch_size * num_proposals, 8, 3
            detected_bbox_corners.view(-1, 8, 3) # batch_size * num_proposals, 8, 3
        ).view(batch_size, num_proposals)
        
        # find good boxes (IoU > threshold)
        good_bbox_masks = ious > min_iou # batch_size, num_proposals

        # dump generated captions
        for batch_id in range(batch_size):
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = dataset.scanrefer[dataset_idx]["scene_id"]
            for prop_id in range(num_proposals):
                if objn_masks[batch_id, prop_id] == 1 and good_bbox_masks[batch_id, prop_id] == 1:
                # if objn_masks[batch_id, prop_id] == 1:
                    object_id = str(detected_object_ids[batch_id, prop_id].item())
                    caption_decoded = decode_caption(captions[batch_id, prop_id], dataset.vocabulary["idx2word"])

                    # print(scene_id, object_id)
                    try:
                        ann_list = list(SCANREFER_ORGANIZED[scene_id][object_id].keys())
                        object_name = SCANREFER_ORGANIZED[scene_id][object_id][ann_list[0]]["object_name"]

                        # store
                        key = "{}|{}|{}".format(scene_id, object_id, object_name)
                        # key = "{}|{}".format(scene_id, object_id)
                        candidates[key] = [caption_decoded]

                        if save_interm:
                            if scene_id not in intermediates: intermediates[scene_id] = {}
                            if object_id not in intermediates[scene_id]: intermediates[scene_id][object_id] = {}

                            intermediates[scene_id][object_id]["object_name"] = object_name
                            intermediates[scene_id][object_id]["box_corner"] = detected_bbox_corners[batch_id, prop_id].cpu().numpy().tolist()
                            intermediates[scene_id][object_id]["description"] = caption_decoded
                            intermediates[scene_id][object_id]["token"] = caption_decoded.split(" ")

                            # attention context
                            # extract attention masks for each object
                            object_attn_weights = data_dict["topdown_attn"][:, :, :num_proposals] # NOTE only consider attention on objects
                            valid_context_masks = data_dict["valid_masks"][:, :, :num_proposals] # NOTE only consider attention on objects

                            cur_valid_context_masks = valid_context_masks[batch_id, prop_id] # num_proposals
                            cur_context_box_corners = detected_bbox_corners[batch_id, cur_valid_context_masks == 1] # X, 8, 3
                            cur_object_attn_weights = object_attn_weights[batch_id, prop_id, cur_valid_context_masks == 1] # X

                            intermediates[scene_id][object_id]["object_attn_weight"] = cur_object_attn_weights.cpu().numpy().T.tolist()
                            intermediates[scene_id][object_id]["object_attn_context"] = cur_context_box_corners.cpu().numpy().tolist()
                    
                    except KeyError:
                        continue

    # detected boxes
    if save_interm:
        print("saving intermediate results...")
        interm_path = os.path.join(CONF.PATH.OUTPUT, folder, "interm.json")
        with open(interm_path, "w") as f:
            json.dump(intermediates, f, indent=4)

    return candidates 

def update_interm(interm, candidates, bleu, cider, rouge, meteor):
    for i, (key, value) in enumerate(candidates.items()):
        scene_id, object_id, object_name = key.split("|")
        if scene_id in interm:
            if object_id in interm[scene_id]:
                interm[scene_id][object_id]["bleu_1"] = bleu[1][0][i]
                interm[scene_id][object_id]["bleu_2"] = bleu[1][1][i]
                interm[scene_id][object_id]["bleu_3"] = bleu[1][2][i]
                interm[scene_id][object_id]["bleu_4"] = bleu[1][3][i]

                interm[scene_id][object_id]["cider"] = cider[1][i]

                interm[scene_id][object_id]["rouge"] = rouge[1][i]

                interm[scene_id][object_id]["meteor"] = meteor[1][i]

    return interm

def eval_cap(mode, model, dataset, dataloader, phase, folder, 
    use_tf=False, max_len=CONF.TRAIN.MAX_DES_LEN, force=False, min_iou=CONF.EVAL.MIN_IOU_THRESHOLD, save_interm=False, no_stop_words=False):
    
    # corpus
    corpus_path = os.path.join(CONF.PATH.OUTPUT, folder, "corpus_{}.json".format(phase))
    if not os.path.exists(corpus_path) or force:
        print("preparing corpus...")
        corpus = prepare_corpus(dataset.scanrefer, max_len)
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=4)
    else:
        print("loading corpus...")
        with open(corpus_path) as f:
            corpus = json.load(f)

    pred_path = os.path.join(CONF.PATH.OUTPUT, folder, "pred_{}.json".format(phase))
    # generate results
    print("generating descriptions...")
    if mode == "gt":
        candidates = feed_gt_cap(model, dataset, dataloader, phase, folder, use_tf, max_len, save_interm)
    elif mode == "votenet":
        candidates = feed_votenet_cap(model, dataset, dataloader, phase, folder, use_tf, max_len, save_interm, min_iou)
    else:
        raise ValueError("invalid mode, choice: [gt, votenet]")

    # check candidates
    # NOTE: make up the captions for the undetected object by "sos eos"
    candidates = check_candidates(corpus, candidates)
    candidates = organize_candidates(corpus, candidates)

    if no_stop_words:
        print("filtering out stops words...")
        corpus, candidates = filter_stop_words(corpus, candidates)

    with open(pred_path, "w") as f:
        json.dump(candidates, f, indent=4)

    # compute scores
    print("computing scores...")
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)

    # save scores
    print("saving scores...")
    score_path = os.path.join(CONF.PATH.OUTPUT, folder, "score_{}.json".format(phase))
    with open(score_path, "w") as f:
        scores = {
            "bleu-1": [float(s) for s in bleu[1][0]],
            "bleu-2": [float(s) for s in bleu[1][1]],
            "bleu-3": [float(s) for s in bleu[1][2]],
            "bleu-4": [float(s) for s in bleu[1][3]],
            "cider": [float(s) for s in cider[1]],
            "rouge": [float(s) for s in rouge[1]],
            "meteor": [float(s) for s in meteor[1]],
        }
        json.dump(scores, f, indent=4)

    # update intermediates
    if save_interm:
        print("updating intermediate results...")
        interm_path = os.path.join(CONF.PATH.OUTPUT, folder, "interm.json")
        with open(interm_path) as f:
            interm = json.load(f)

        interm = update_interm(interm, candidates, bleu, cider, rouge, meteor)

        with open(interm_path, "w") as f:
            json.dump(interm, f, indent=4)

    return bleu, cider, rouge, meteor

