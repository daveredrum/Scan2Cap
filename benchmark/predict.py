# HACK ignore warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
from lib.dataset import ScannetReferenceDataset
from lib.config import CONF
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_scene_cap_loss
from models.capnet import CapNet
from lib.eval_helper import eval_cap

## constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, config):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list,  
        split="val",
        name=args.dataset,
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        augment=False
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return dataset, dataloader

def get_model(args, dataset, device, root=CONF.PATH.OUTPUT, eval_pretrained=False):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = CapNet(
        num_class=DC.num_class,
        vocabulary=dataset.vocabulary,
        embeddings=dataset.glove,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        no_caption=not args.eval_caption,
        use_topdown=args.use_topdown,
        num_locals=args.num_locals,
        query_mode=args.query_mode,
        graph_mode=args.graph_mode,
        num_graph_steps=args.num_graph_steps,
        use_relation=args.use_relation
    )

    if eval_pretrained:
        # load pretrained model
        print("loading pretrained VoteNet...")
        pretrained_model = CapNet(
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

        pretrained_name = "PRETRAIN_VOTENET_XYZ"
        if args.use_color: pretrained_name += "_COLOR"
        if args.use_multiview: pretrained_name += "_MULTIVIEW"
        if args.use_normal: pretrained_name += "_NORMAL"

        pretrained_path = os.path.join(CONF.PATH.PRETRAINED, pretrained_name, "model.pth")
        pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

        # mount
        model.backbone_net = pretrained_model.backbone_net
        model.vgen = pretrained_model.vgen
        model.proposal = pretrained_model.proposal
    else:
        # load
        model_name = "model_last.pth" if args.use_last else "model.pth"
        model_path = os.path.join(root, args.folder, model_name)
        model.load_state_dict(torch.load(model_path), strict=False)
        # model.load_state_dict(torch.load(model_path))
    
    # to device
    model.to(device)

    # set mode
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    scene_list = [s for s in scene_list if s.split("_")[-1] == "00"]

    return scene_list

def get_eval_data(args):
    scanrefer_test = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_{}.json".format(args.test_split))))

    eval_scene_list = get_scannet_scene_list(args.test_split)
    scanrefer_eval = []
    for scene_id in eval_scene_list:
        data = deepcopy(scanrefer_test[0])
        data["scene_id"] = scene_id
        scanrefer_eval.append(data)

    print("test on {} samples".format(len(scanrefer_eval)))

    return scanrefer_eval, eval_scene_list

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

def predict_caption(args, root=CONF.PATH.OUTPUT):
    print("initializing...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get eval data
    scanrefer_eval, eval_scene_list = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_dataloader(args, scanrefer_eval, eval_scene_list, DC)

    # get model
    model = get_model(args, dataset, device)

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

    # evaluate
    print("generating results...")
    outputs = {}
    for data_dict in tqdm(dataloader):
        # to device
        for key in data_dict:
            if isinstance(data_dict[key], list): continue
            data_dict[key] = data_dict[key].cuda()

        # feed
        with torch.no_grad():
            data_dict = model(data_dict, False, True)
            data_dict = get_scene_cap_loss(data_dict, device, DC, weights=dataset.weights, detection=True, caption=False)

        # unpack
        pred_captions = data_dict["lang_cap"].argmax(-1) # batch_size, num_proposals, max_len - 1
        dataset_ids = data_dict["dataset_idx"]
        pred_boxes = data_dict["bbox_corner"]

        # nms mask
        _ = parse_predictions(data_dict, POST_DICT)
        nms_masks = torch.FloatTensor(data_dict["pred_mask"]).type_as(pred_boxes).long()

        # objectness mask
        obj_masks = data_dict["bbox_mask"]

        # final mask
        nms_masks = nms_masks * obj_masks

        # nms_masks = torch.ones(pred_boxes.shape[0], pred_boxes.shape[1]).type_as(pred_boxes)

        # for object detection
        pred_sem_prob = torch.softmax(data_dict['sem_cls_scores'], dim=-1) # B, num_proposal, num_cls
        pred_obj_prob = torch.softmax(data_dict['objectness_scores'], dim=-1) # B, num_proposal, 2

        for batch_id in range(pred_captions.shape[0]):
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = dataset.scanrefer[dataset_idx]["scene_id"]
            scene_outputs = []
            for object_id in range(pred_captions.shape[1]):
                if nms_masks[batch_id, object_id] == 1:
                    caption = decode_caption(pred_captions[batch_id, object_id], dataset.vocabulary["idx2word"]) 
                    box = pred_boxes[batch_id, object_id].cpu().detach().numpy().tolist()

                    sem_prob = pred_sem_prob[batch_id, object_id].cpu().detach().numpy().tolist()
                    obj_prob = pred_obj_prob[batch_id, object_id].cpu().detach().numpy().tolist()

                    scene_outputs.append(
                        {
                            "caption": caption,
                            "box": box,
                            "sem_prob": sem_prob,
                            "obj_prob": obj_prob
                        }
                    )

            outputs[scene_id] = scene_outputs

    # dump
    save_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "pred.json")
    with open(save_path, "w") as f:
        json.dump(outputs, f, indent=4)

    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer or ReferIt3D", default="ScanRefer")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    # parser.add_argument("--gpu", type=str, help="gpu", default=["0"], nargs="+")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--num_locals", type=int, default=-1, help="Number of local objects [default: -1]")
    parser.add_argument("--num_graph_steps", type=int, default=0, help="Number of graph conv layer [default: 0]")
    
    parser.add_argument("--query_mode", type=str, default="corner", help="Mode for querying the local context, [choices: center, corner]")
    parser.add_argument("--graph_mode", type=str, default="edge_conv", help="Mode for querying the local context, [choices: graph_conv, edge_conv]")
    parser.add_argument("--graph_aggr", type=str, default="add", help="Mode for aggregating features, [choices: add, mean, max]")
    
    parser.add_argument("--min_iou", type=float, default=0.25, help="Min IoU threshold for evaluation")
    
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    
    parser.add_argument("--use_tf", action="store_true", help="Enable teacher forcing")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    parser.add_argument("--use_last", action="store_true", help="Use the last model")
    parser.add_argument("--use_topdown", action="store_true", help="Use top-down attention for captioning.")
    parser.add_argument("--use_relation", action="store_true", help="Use object-to-object relation in graph.")
    
    parser.add_argument("--eval_caption", action="store_true", help="evaluate the reference localization results")
    parser.add_argument("--eval_detection", action="store_true", help="evaluate the object detection results")
    parser.add_argument("--eval_pretrained", action="store_true", help="evaluate the pretrained object detection results")
    
    parser.add_argument("--force", action="store_true", help="generate the results by force")
    parser.add_argument("--save_interm", action="store_true", help="Save the intermediate results")
    
    parser.add_argument("--test_split", type=str, default="test", help="Mode for aggregating features, [choices: train, val, test]")
    
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # evaluate
    predict_caption(args)

