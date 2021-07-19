# HACK ignore warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.solver_pretrained import Solver
from lib.config import CONF
from models.capnet_pretrained import CapNet

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
# SCANREFER_DUMMY = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_dummy.json")))

# extracted ScanNet object rotations from Scan2CAD 
# NOTE some scenes are missing in this annotation!!!
SCAN2CAD_ROTATION = json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json")))

# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, split, augment, scan2cad_rotation=None):
    if args.mode == "gt":
        from lib.dataset_pretrained import PretrainedGTDataset as PretrainedDataset
    elif args.mode == "votenet":
        from lib.dataset_pretrained import PretrainedVoteNetDataset as PretrainedDataset
    else:
        raise ValueError("invalid pretrained mode, choices: [gt, votenet]")

    dataset = PretrainedDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        augment=augment,
        debug=args.debug,
        scan2cad_rotation=scan2cad_rotation
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return dataset, dataloader

def get_model(args, dataset):
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
        graph_aggr=args.graph_aggr,
        use_relation=args.use_relation,
        use_orientation=args.use_orientation,
        use_distance=args.use_distance
    )
    
    # to device
    model.cuda()

    return model

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataset, dataloader):
    model = get_model(args, dataset["train"])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    LR_DECAY_STEP = [80, 120, 160] if args.use_scheduler else None
    LR_DECAY_RATE = 0.1 if args.use_scheduler else None
    BN_DECAY_STEP = 20 if args.use_scheduler else None
    BN_DECAY_RATE = 0.5 if args.use_scheduler else None

    solver = Solver(
        mode=args.mode,
        model=model, 
        config=DC, 
        dataset=dataset,
        dataloader=dataloader, 
        optimizer=optimizer, 
        stamp=stamp, 
        val_step=args.val_step,
        use_tf=not args.no_tf,
        use_orientation=args.use_orientation,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE,
        criterion=args.criterion
    )
    num_params = get_num_params(model)

    return solver, num_params, root

def save_info(args, root, num_params, dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = len(dataset["train"])
    info["num_eval_train"] = len(dataset["eval"]["train"])
    info["num_eval_val"] = len(dataset["eval"]["val"])
    info["num_train_scenes"] = len(dataset["train"].scene_list)
    info["num_eval_train_scenes"] = len(dataset["eval"]["train"].scene_list)
    info["num_eval_val_scenes"] = len(dataset["eval"]["val"].scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def get_scanrefer(args):
    train_scene_list = sorted(list(set([data["scene_id"] for data in SCANREFER_TRAIN])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in SCANREFER_VAL])))

    scanrefer_train = SCANREFER_TRAIN

    # eval
    scanrefer_eval_train = []
    for scene_id in train_scene_list:
        data = deepcopy(SCANREFER_TRAIN[0])
        data["scene_id"] = scene_id
        scanrefer_eval_train.append(data)
    
    scanrefer_eval_val = []
    for scene_id in val_scene_list:
        data = deepcopy(SCANREFER_TRAIN[0])
        data["scene_id"] = scene_id
        scanrefer_eval_val.append(data)

    print("train on {} samples from {} scenes".format(len(scanrefer_eval_train), len(train_scene_list)))
    print("eval on {} scenes from train and {} scenes from val".format(len(train_scene_list), len(val_scene_list)))

    return scanrefer_train, scanrefer_eval_train, scanrefer_eval_val, train_scene_list, val_scene_list

def train(args):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_eval_train, scanrefer_eval_val, train_scene_list, val_scene_list = get_scanrefer(args)

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer_train, train_scene_list, "train", not args.no_augment, SCAN2CAD_ROTATION)
    eval_train_dataset, eval_train_dataloader = get_dataloader(args, scanrefer_eval_train, train_scene_list, "train", False)
    eval_val_dataset, eval_val_dataloader = get_dataloader(args, scanrefer_eval_val, val_scene_list, "val", False)
    dataset = {
        "train": train_dataset,
        "eval": {
            "train": eval_train_dataset,
            "val": eval_val_dataset
        }
    }
    dataloader = {
        "train": train_dataloader,
        "eval": {
            "train": eval_train_dataloader,
            "val": eval_val_dataloader
        }
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataset, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, dataset)
    solver(args.epoch, args.verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--mode", type=str, help="Mode for pretrained pipeline, choice: [gt, votenet]", required=True)
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=1000)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=2000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--num_locals", type=int, default=-1, help="Number of local objects [default: -1]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--num_graph_steps", type=int, default=0, help="Number of graph conv layer [default: 0]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--criterion", type=str, default="cider", \
        help="criterion for selecting the best model [choices: bleu-1, bleu-2, bleu-3, bleu-4, cider, rouge, meteor, sum]")
    parser.add_argument("--query_mode", type=str, default="center", help="Mode for querying the local context, [choices: center, corner]")
    parser.add_argument("--graph_mode", type=str, default="edge_conv", help="Mode for querying the local context, [choices: graph_conv, edge_conv]")
    parser.add_argument("--graph_aggr", type=str, default="add", help="Mode for aggregating features, [choices: add, mean, max]")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_tf", action="store_true", help="Do NOT enable teacher forcing in inference.")
    parser.add_argument("--use_topdown", action="store_true", help="Use top-down attention for captioning.")
    parser.add_argument("--use_relation", action="store_true", help="Use object-to-object relation in graph.")
    parser.add_argument("--use_orientation", action="store_true", help="Use object-to-object orientation loss in graph.")
    parser.add_argument("--use_distance", action="store_true", help="Use object-to-object distance loss in graph.")
    parser.add_argument("--use_scheduler", action="store_true", help="Use LR scheduler for training.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)
    
