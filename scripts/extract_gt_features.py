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
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig

from lib.dataset_maskvotenet import MaskScannetReferenceDataset
from lib.config import CONF

from models.mask_votenet import MaskVoteNet

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
# SCANREFER_DUMMY = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_dummy.json")))

# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, split, augment):
    dataset = MaskScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        augment=augment
    )

    print("using {} samples in the {} split".format(len(dataset), split))

    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return dataset, dataloader

def get_model(args):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height) + 1
    model = MaskVoteNet(
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=1
    )

    model_name = "XYZ"
    if args.use_color: model_name += "_COLOR"
    if args.use_multiview: model_name += "_MULTIVIEW"
    if args.use_normal: model_name += "_NORMAL"
    model_name += "_MASK_VOTENET"

    pretrained_path = os.path.join(CONF.PATH.PRETRAINED, model_name, "model.pth")
    model.load_state_dict(torch.load(pretrained_path), strict=False)

    # to CUDA
    model.cuda()

    # set mode
    model.eval()

    return model

def get_scanrefer():
    scanrefer_train = SCANREFER_TRAIN
    scanrefer_val = SCANREFER_VAL

    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    return scanrefer_train, scanrefer_val, all_scene_list

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_{}.txt".format(split)))])

    return scene_list

def extract_train(args, model, dataset, dataloader):
    print("extracting features for the train split...")
    os.makedirs(CONF.PATH.GT_FEATURES.format(dataset.name), exist_ok=True)
    database = h5py.File(os.path.join(CONF.PATH.GT_FEATURES.format(dataset.name), "train.hdf5"), "w", libver="latest")
    for epoch_id in range(args.epoch):
        print("extracting for epoch {}...".format(epoch_id))
       
        # HACK temporarily storing all results in a dict, then dump into hdf5 database after organization
        epoch_dict = {}

        for data in tqdm(dataloader):
            # move to CUDA
            for key in data:
                data[key] = data[key].cuda()

            # fead
            with torch.no_grad():
                data = model(data)

            # unpack
            dataset_ids = data["dataset_idx"]
            features = data["aggregated_vote_features"][:, 0]
            
            bbox_corners = data["bbox_corner"]

            # store
            batch_size = dataset_ids.shape[0]

            for batch_id in range(batch_size):
                # info
                dataset_idx = dataset_ids[batch_id].item()
                scene_id = dataset.scanrefer[dataset_idx]["scene_id"]
                object_id = int(dataset.scanrefer[dataset_idx]["object_id"])

                # features
                cur_feat = features[batch_id]
                cur_corners = bbox_corners[batch_id]

                # save to dict
                if scene_id not in epoch_dict: 
                    epoch_dict[scene_id] = {
                        "object_ids": [],
                        "features": [],
                        "bbox_corners": []
                    }

                epoch_dict[scene_id]["object_ids"].append(object_id)
                epoch_dict[scene_id]["features"].append(cur_feat.cpu().numpy())
                epoch_dict[scene_id]["bbox_corners"].append(cur_corners.cpu().numpy())

        # aggregate epoch data
        for scene_id in epoch_dict.keys():
            # save scene object ids
            object_id_dataset = "{}|{}_gt_ids".format(str(epoch_id), scene_id)
            object_ids = np.array(epoch_dict[scene_id]["object_ids"])
            database.create_dataset(object_id_dataset, data=object_ids)

            # save features
            feature_dataset = "{}|{}_features".format(str(epoch_id), scene_id)
            features = np.stack(epoch_dict[scene_id]["features"], axis=0)
            database.create_dataset(feature_dataset, data=features)

            # save bboxes
            bbox_dataset = "{}|{}_bbox_corners".format(str(epoch_id), scene_id)
            bbox_corners = np.stack(epoch_dict[scene_id]["bbox_corners"], axis=0)
            database.create_dataset(bbox_dataset, data=bbox_corners)
            
            # save GT bboxes
            gt_dataset = "{}|{}_gt_corners".format(str(epoch_id), scene_id)
            gt_corners = np.stack(epoch_dict[scene_id]["bbox_corners"], axis=0)
            database.create_dataset(gt_dataset, data=gt_corners)


def extract_val(args, model, dataset, dataloader):
    print("extracting features for the val split...")
    database = h5py.File(os.path.join(CONF.PATH.GT_FEATURES.format(dataset.name), "val.hdf5"), "w", libver="latest")

    # HACK temporarily storing all results in a dict, then dump into hdf5 database after organization
    epoch_dict = {}

    for data in tqdm(dataloader):
        # move to CUDA
        for key in data:
            data[key] = data[key].cuda()

        # fead
        with torch.no_grad():
            data = model(data)

        # unpack
        dataset_ids = data["dataset_idx"]
        features = data["aggregated_vote_features"][:, 0]
        
        bbox_corners = data["bbox_corner"]

        # store
        batch_size = dataset_ids.shape[0]
        for batch_id in range(batch_size):
            # info
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = dataset.scanrefer[dataset_idx]["scene_id"]
            object_id = int(dataset.scanrefer[dataset_idx]["object_id"])

            # features
            cur_feat = features[batch_id]
            cur_corners = bbox_corners[batch_id]

            # save to dict
            if scene_id not in epoch_dict: 
                epoch_dict[scene_id] = {
                    "object_ids": [],
                    "features": [],
                    "bbox_corners": []
                }

            epoch_dict[scene_id]["object_ids"].append(object_id)
            epoch_dict[scene_id]["features"].append(cur_feat.cpu().numpy())
            epoch_dict[scene_id]["bbox_corners"].append(cur_corners.cpu().numpy())

    # aggregate epoch data
    epoch_id = 0
    for scene_id in epoch_dict.keys():
        # save scene object ids
        object_id_dataset = "{}|{}_gt_ids".format(str(epoch_id), scene_id)
        object_ids = np.array(epoch_dict[scene_id]["object_ids"])
        database.create_dataset(object_id_dataset, data=object_ids)

        # save features
        feature_dataset = "{}|{}_features".format(str(epoch_id), scene_id)
        features = np.stack(epoch_dict[scene_id]["features"], axis=0)
        database.create_dataset(feature_dataset, data=features)

        # save bboxes
        bbox_dataset = "{}|{}_bbox_corners".format(str(epoch_id), scene_id)
        bbox_corners = np.stack(epoch_dict[scene_id]["bbox_corners"], axis=0)
        database.create_dataset(bbox_dataset, data=bbox_corners)

        # save GT bboxes
        gt_dataset = "{}|{}_gt_corners".format(str(epoch_id), scene_id)
        gt_corners = np.stack(epoch_dict[scene_id]["bbox_corners"], axis=0)
        database.create_dataset(gt_dataset, data=gt_corners)

def extract(args):
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer()

    # dataloader
    print("initializing dataloader...")
    train_dataset, train_dataloader = get_dataloader(args, scanrefer_train, all_scene_list, "train", True)
    val_dataset, val_dataloader = get_dataloader(args, scanrefer_val, all_scene_list, "val", False)

    # get model
    print("initializing the model...")
    model = get_model(args)

    # extract features from train split
    # NOTE also extracting features for different epoch
    if args.train:
        extract_train(args, model, train_dataset, train_dataloader)

    # extract features from val split
    # NOTE only extract for once
    if args.val:
        extract_val(args, model, val_dataset, val_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    # parser.add_argument("--num_points", type=int, default=1024, help="Point Number [default: 1024]")
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    # parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--train", action="store_true", help="Save features for the train split.")
    parser.add_argument("--val", action="store_true", help="Save features for the val split.")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    extract(args)
