import os
import sys
import json
import torch
import argparse

import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge

from data.scannet.model_util_scannet import ScannetDatasetConfig

from lib.config import CONF
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_scene_cap_loss
from models.capnet_pretrained import CapNet
from lib.eval_helper_pretrained import eval_cap

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
# SCANREFER_DUMMY = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_dummy.json")))

# extracted ScanNet object rotations from Scan2CAD 
# NOTE some scenes are missing in this annotation!!!
SCAN2CAD_ROTATION = json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json")))

# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, config):
    if args.mode == "gt":
        from lib.dataset_pretrained import PretrainedGTDataset as PretrainedDataset
    elif args.mode == "votenet":
        from lib.dataset_pretrained import PretrainedVoteNetDataset as PretrainedDataset
    else:
        raise ValueError("invalid pretrained mode, choices: [gt, votenet]")

    dataset = PretrainedDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list, 
        split="val",
        scan2cad_rotation=SCAN2CAD_ROTATION
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return dataset, dataloader

def get_model(args, dataset, device, root=CONF.PATH.OUTPUT):
    # initiate model
    model = CapNet(
        mode=args.mode,
        vocabulary=dataset.vocabulary,
        embeddings=dataset.glove,
        use_topdown=args.use_topdown,
        num_locals=args.num_locals,
        query_mode=args.query_mode,
        graph_mode=args.graph_mode,
        num_graph_steps=args.num_graph_steps,
        use_relation=args.use_relation,
        use_orientation=args.use_orientation,
        graph_aggr=args.graph_aggr
    )

    # load
    model_name = "model_last.pth" if args.use_last else "model.pth"
    model_path = os.path.join(root, args.folder, model_name)
    # model.load_state_dict(torch.load(model_path), strict=False)
    model.load_state_dict(torch.load(model_path))
    
    # to device
    model.to(device)

    # set mode
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_{}.txt".format(split)))])

    return scene_list

def get_eval_data(args):
    eval_scene_list = get_scannet_scene_list("train") if args.use_train else get_scannet_scene_list("val")
    scanrefer_eval = []
    for scene_id in eval_scene_list:
        data = deepcopy(SCANREFER_TRAIN[0]) if args.use_train else deepcopy(SCANREFER_VAL[0])
        data["scene_id"] = scene_id
        scanrefer_eval.append(data)

    print("eval on {} samples".format(len(scanrefer_eval)))

    return scanrefer_eval, eval_scene_list

def evaluate(args):
    print("initializing...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get eval data
    scanrefer_eval, eval_scene_list = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_dataloader(args, scanrefer_eval, eval_scene_list, DC)

    # get model
    model = get_model(args, dataset, device)

    # evaluate
    bleu, cider, rouge, meteor = eval_cap(args.mode, model, dataset, dataloader, "val", args.folder, args.use_tf, min_iou=args.min_iou, save_interm=args.save_interm)

    # report
    print("\n----------------------Evaluation-----------------------")
    print("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][0], max(bleu[1][0]), min(bleu[1][0])))
    print("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][1], max(bleu[1][1]), min(bleu[1][1])))
    print("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][2], max(bleu[1][2]), min(bleu[1][2])))
    print("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][3], max(bleu[1][3]), min(bleu[1][3])))
    print("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(cider[0], max(cider[1]), min(cider[1])))
    print("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(rouge[0], max(rouge[1]), min(rouge[1])))
    print("[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(meteor[0], max(meteor[1]), min(meteor[1])))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--mode", type=str, help="Mode for pretrained pipeline, choice: [gt, votenet]", required=True)
    # parser.add_argument("--gpu", type=str, help="gpu", default="0")
    # parser.add_argument("--gpu", type=str, help="gpu", default=["0"], nargs="+")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--num_locals", type=int, default=-1, help="Number of local objects [default: -1]")
    parser.add_argument("--num_graph_steps", type=int, default=0, help="Number of graph conv layer [default: 0]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    parser.add_argument("--use_tf", action="store_true", help="Enable teacher forcing")
    parser.add_argument("--query_mode", type=str, default="center", help="Mode for querying the local context, [choices: center, corner]")
    parser.add_argument("--graph_mode", type=str, default="edge_conv", help="Mode for querying the local context, [choices: graph_conv, edge_conv]")
    parser.add_argument("--graph_aggr", type=str, default="add", help="Mode for aggregating features, [choices: add, mean, max]")
    parser.add_argument("--min_iou", type=float, default=0.25, help="Min IoU threshold for evaluation")
    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    parser.add_argument("--use_last", action="store_true", help="Use the last model")
    parser.add_argument("--use_topdown", action="store_true", help="Use top-down attention for captioning.")
    parser.add_argument("--use_relation", action="store_true", help="Use object-to-object relation in graph.")
    parser.add_argument("--use_orientation", action="store_true", help="Use object-to-object orientation loss in graph.")
    parser.add_argument("--force", action="store_true", help="generate the results by force")
    parser.add_argument("--save_interm", action="store_true", help="Save the intermediate results")
    args = parser.parse_args()

    # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # evaluate
    evaluate(args)

