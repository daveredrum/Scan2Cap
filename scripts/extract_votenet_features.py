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
from tqdm import tqdm
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.config import CONF
from lib.ap_helper import parse_predictions
from lib.loss_helper import get_scene_cap_loss
from models.capnet import CapNet
from utils.box_util import get_3d_box_batch

# constants
DC = ScannetDatasetConfig()
# SCANREFER_DUMMY = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_dummy.json")))

def get_dataloader(args, scanrefer, all_scene_list, split, augment):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list,  
        split=split,
        name=args.dataset,
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

def get_model(args, dataset):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    
    model = CapNet(
        num_class=DC.num_class,
        vocabulary=dataset.vocabulary,
        embeddings=dataset.glove,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        num_proposal=args.num_proposals,
        input_feature_dim=input_channels,
        no_caption=True
    )

    # pretrained_path = os.path.join(CONF.PATH.PRETRAINED, "XYZ_COLOR_NORMAL_DETECTION", "model.pth")

    model_root = "PRETRAIN_VOTENET_XYZ"
    if args.use_color: model_root += "_COLOR"
    if args.use_multiview: model_root += "_MULTIVIEW"
    if args.use_normal: model_root += "_NORMAL"

    pretrained_path = os.path.join(CONF.PATH.PRETRAINED, model_root, "model.pth")
    model.load_state_dict(torch.load(pretrained_path), strict=False)

    # to CUDA
    model.cuda()

    # set mode
    model.eval()

    return model

def get_scanrefer():
    if args.dataset == "ScanRefer":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
        scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
    elif args.dataset == "ReferIt3D":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
        scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_val.json")))
    else:
        raise ValueError("Invalid dataset.")

    train_scene_list = get_scannet_scene_list(scanrefer_train)
    val_scene_list = get_scannet_scene_list(scanrefer_val)

    DUMMY = [scanrefer_train[0]]

    scanrefer_train = []
    for scene_id in train_scene_list:
        data = deepcopy(DUMMY)
        data["scene_id"] = scene_id
        scanrefer_train.append(data)

    scanrefer_val = []
    for scene_id in val_scene_list:
        data = deepcopy(DUMMY)
        data["scene_id"] = scene_id
        scanrefer_val.append(data)

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    return scanrefer_train, scanrefer_val, all_scene_list

def get_scannet_scene_list(scanrefer):
    scene_list = sorted(list(set([d["scene_id"] for d in scanrefer])))

    return scene_list

def extract_train(args, model, dataset, dataloader, post_dict):
    print("extracting features for the train split...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONF.PATH.VOTENET_FEATURES.format(dataset.name), exist_ok=True)
    database = h5py.File(os.path.join(CONF.PATH.VOTENET_FEATURES.format(dataset.name), "train.hdf5"), "w", libver="latest")
    for epoch_id in range(args.epoch):
        print("extracting for epoch {}...".format(epoch_id))
        for data in tqdm(dataloader):
            # move to CUDA
            for key in data:
                data[key] = data[key].cuda()

            # fead
            with torch.no_grad():
                data = model(data)
                data = get_scene_cap_loss(data, device, DC, weights=dataset.weights, detection=True, caption=False)

            # post-process
            _ = parse_predictions(data, post_dict)
            nms_masks = torch.LongTensor(data["pred_mask"]).cuda()

            # unpack
            dataset_ids = data["dataset_idx"]
            features = data["bbox_feature"]
            bbox_corners = data["bbox_corner"]
            objn_masks = data["bbox_mask"]
            point_clouds = data["point_clouds"][:, :, :3] # only the geometry
            gt_bbox_corners = data["gt_box_corner_label"]
            gt_box_masks = data["gt_box_masks"]
            gt_box_object_ids = data["gt_box_object_ids"]

            # pick out object ids of detected objects
            detected_object_ids = torch.gather(data["scene_object_ids"], 1, data["object_assignment"])

            # store
            batch_size, num_proposals, _, _ = bbox_corners.shape
            for batch_id in range(batch_size):
                # info
                dataset_idx = dataset_ids[batch_id].item()
                scene_id = dataset.scanrefer[dataset_idx]["scene_id"]
                scene_object_ids = []
                scene_feats = []
                scene_corners = []
                for prop_id in range(num_proposals):
                    if objn_masks[batch_id, prop_id] == 1 and nms_masks[batch_id, prop_id] == 1:
                        # detected object id
                        cur_object_id = detected_object_ids[batch_id, prop_id].item()

                        # features
                        cur_feat = features[batch_id, prop_id] # 128
                        cur_corners = bbox_corners[batch_id, prop_id] # 8, 3

                        # append
                        scene_object_ids.append(cur_object_id)
                        scene_feats.append(cur_feat.unsqueeze(0).cpu().numpy())
                        scene_corners.append(cur_corners.unsqueeze(0).cpu().numpy())

                # save scene object ids
                object_id_dataset = "{}|{}_object_ids".format(str(epoch_id), scene_id)
                database.create_dataset(object_id_dataset, data=np.array(scene_object_ids))
                
                # save features
                feature_dataset = "{}|{}_features".format(str(epoch_id), scene_id)
                database.create_dataset(feature_dataset, data=np.concatenate(scene_feats, axis=0))

                # save features
                corner_dataset = "{}|{}_bbox_corners".format(str(epoch_id), scene_id)
                database.create_dataset(corner_dataset, data=np.concatenate(scene_corners, axis=0))

                # save scene object ids
                object_id_dataset = "{}|{}_gt_ids".format(str(epoch_id), scene_id)
                batch_gt_ids = gt_box_object_ids[batch_id, gt_box_masks[batch_id] == 1].cpu().numpy()
                database.create_dataset(object_id_dataset, data=batch_gt_ids)

                # save GT bboxes
                gt_dataset = "{}|{}_gt_corners".format(str(epoch_id), scene_id)
                batch_gt_corners = gt_bbox_corners[batch_id, gt_box_masks[batch_id] == 1].cpu().numpy()
                database.create_dataset(gt_dataset, data=batch_gt_corners)

                # # save point clouds
                # pc_dataset = "{}|{}_pc".format(str(epoch_id), scene_id)
                # database.create_dataset(pc_dataset, data=point_clouds[batch_id].cpu().numpy())

def extract_val(args, model, dataset, dataloader, post_dict):
    print("extracting features for the val split...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    database = h5py.File(os.path.join(CONF.PATH.VOTENET_FEATURES.format(dataset.name), "val.hdf5"), "w", libver="latest")
    for data in tqdm(dataloader):
        # move to CUDA
        for key in data:
            data[key] = data[key].cuda()

        # fead
        with torch.no_grad():
            data = model(data)
            data = get_scene_cap_loss(data, device, DC, weights=dataset.weights, detection=True, caption=False)

        # post-process
        _ = parse_predictions(data, post_dict)
        nms_masks = torch.LongTensor(data["pred_mask"]).cuda()

        # unpack
        dataset_ids = data["dataset_idx"]
        features = data["bbox_feature"]
        bbox_corners = data["bbox_corner"]
        objn_masks = data["bbox_mask"]
        point_clouds = data["point_clouds"][:, :, :3] # only the geometry
        gt_bbox_corners = data["gt_box_corner_label"]
        gt_box_masks = data["gt_box_masks"]
        gt_box_object_ids = data["gt_box_object_ids"]

        # pick out object ids of detected objects
        detected_object_ids = torch.gather(data["scene_object_ids"], 1, data["object_assignment"])

        # store
        batch_size, num_proposals, _, _ = bbox_corners.shape
        for batch_id in range(batch_size):
            # info
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = dataset.scanrefer[dataset_idx]["scene_id"]
            scene_object_ids = []
            scene_feats = []
            scene_corners = []
            for prop_id in range(num_proposals):
                if objn_masks[batch_id, prop_id] == 1 and nms_masks[batch_id, prop_id] == 1:
                    # detected object id
                    cur_object_id = detected_object_ids[batch_id, prop_id].item()

                    # features
                    cur_feat = features[batch_id, prop_id] # 128
                    cur_corners = bbox_corners[batch_id, prop_id] # 8, 3

                    # append
                    scene_object_ids.append(cur_object_id)
                    scene_feats.append(cur_feat.unsqueeze(0).cpu().numpy())
                    scene_corners.append(cur_corners.unsqueeze(0).cpu().numpy())

            # save scene object ids
            object_id_dataset = "0|{}_object_ids".format(scene_id)
            database.create_dataset(object_id_dataset, data=np.array(scene_object_ids))
            
            # save features
            feature_dataset = "0|{}_features".format(scene_id)
            database.create_dataset(feature_dataset, data=np.concatenate(scene_feats, axis=0))

            # save features
            corner_dataset = "0|{}_bbox_corners".format(scene_id)
            database.create_dataset(corner_dataset, data=np.concatenate(scene_corners, axis=0))

            # save scene object ids
            object_id_dataset = "0|{}_gt_ids".format(scene_id)
            batch_gt_ids = gt_box_object_ids[batch_id, gt_box_masks[batch_id] == 1].cpu().numpy()
            database.create_dataset(object_id_dataset, data=batch_gt_ids)

            # save GT bboxes
            gt_dataset = "0|{}_gt_corners".format(scene_id)
            batch_gt_corners = gt_bbox_corners[batch_id, gt_box_masks[batch_id] == 1].cpu().numpy()
            database.create_dataset(gt_dataset, data=batch_gt_corners)

            # # save point clouds
            # pc_dataset = "0|{}_pc".format(scene_id)
            # database.create_dataset(pc_dataset, data=point_clouds[batch_id].cpu().numpy())

def extract(args):
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer()

    # dataloader
    print("initializing dataloader...")
    train_dataset, train_dataloader = get_dataloader(args, scanrefer_train, all_scene_list, "train", True)
    val_dataset, val_dataloader = get_dataloader(args, scanrefer_val, all_scene_list, "val", False)

    # get model
    print("initializing the model...")
    model = get_model(args, train_dataset)

    # post-process
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

    # extract features from train split
    # NOTE also extracting features for different epoch
    if args.train:
        extract_train(args, model, train_dataset, train_dataloader, POST_DICT)

    # extract features from val split
    # NOTE only extract for once
    if args.val:
        extract_val(args, model, val_dataset, val_dataloader, POST_DICT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--epoch", type=int, help="number of epochs", default=100)
    parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer or ReferIt3D", default="ScanRefer")
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--train", action="store_true", help="Save features for the train split.")
    parser.add_argument("--val", action="store_true", help="Save features for the val split.")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    extract(args)
