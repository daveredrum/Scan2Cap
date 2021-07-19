import os
import torch
import argparse
import numpy as np
from lib.conf import get_config, get_samples
from preprocessing.utils import export_bbox_pickle_raw, export_bbox_pickle_coco, export_image_features, \
    export_bbox_features
from scripts.train import train_main
from scripts.eval import eval_main
import random


def parse_arg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_type", default="nret", help="retrieval or nonretrieval")
    ap.add_argument("--dataset", default="scanrefer", help="scanrefer or referit")
    ap.add_argument("--viewpoint", default="annotated", help="annotated, estimated or bev")
    ap.add_argument("--box", default="oracle", help="oracle, mrcnn or votenet")

    ap.add_argument("--prep", action="store_true", default=False)
    ap.add_argument("--train", action="store_true", default=False)
    ap.add_argument("--model", type=str, default="snt", help='satnt or snt')
    ap.add_argument("--visual_feat", type=str, default='G')
    ap.add_argument("--eval", action="store_true", default=False)

    ap.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    ap.add_argument("--gpu", type=str, help="gpu", default="0")
    ap.add_argument("--batch_size", type=int, help="batch size", default=64)
    ap.add_argument("--num_epochs", type=int, help="number of epochs", default=50)
    ap.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--val_step", type=int, help="iterations of validating", default=2000)
    ap.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    ap.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    ap.add_argument("--seed", type=int, default=3, help="random seed")
    ap.add_argument("--folder", type=str, required=False)
    ap.add_argument("--shuffle", action='store_true', default=True)
    ap.add_argument("--ckpt_path", type=str, default="")
    ap.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    ap.add_argument("--extras", action="store_true", default=False)
    return ap.parse_args()


def prep_main(exp_type, dataset, viewpoint, box):
    run_config = get_config(exp_type, dataset, viewpoint, box)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for the given dataset, viewpoint and box mode
    # performs the following:
    # 1. Load proper config
    # 2. Extract global ResNet101 features
    # 3. Extract bounding boxes from aggregations and instance masks
    # 4. Extract bounding box features

    # 2. Run on CPU
    if box == 'mrcnn':
        sample_list, scene_list = get_samples(mode='val', key_type=run_config.TYPES.KEY_TYPE)
        export_bbox_pickle_coco(
            MRCNN_DETECTIONS_PATH=run_config.PATH.MRCNN_DETECTIONS_PATH,
            DB_PATH=run_config.PATH.DB_PATH,
            GT_DB_PATH=run_config.PATH.GT_DB_PATH, # used for IoU calculation
            RESIZE=(run_config.SCAN_WIDTH, run_config.SCAN_HEIGHT)
        )

    elif box == 'oracle' or 'votenet':
        sample_list, scene_list = get_samples(mode='all', key_type=run_config.TYPES.KEY_TYPE)
        export_bbox_pickle_raw(
            AGGR_JSON_PATH=run_config.PATH.AGGR_JSON,
            SCANNET_V2_TSV=run_config.PATH.SCANNET_V2_TSV,
            INSTANCE_MASK_PATH=run_config.PATH.INSTANCE_MASK,
            SAMPLE_LIST=sample_list,
            SCENE_LIST=scene_list,
            DB_PATH=run_config.PATH.DB_PATH,
            RESIZE=(run_config.SCAN_WIDTH, run_config.SCAN_HEIGHT)
        )
    else:
        raise NotImplementedError('Box mode {} is not implemented.'.format(box))
        
    # 3. Run on Device
    export_image_features(
        KEY_FORMAT=run_config.TYPES.KEY_FORMAT,
        RESIZE=(run_config.SCAN_WIDTH, run_config.SCAN_HEIGHT),
        IMAGE=run_config.PATH.IMAGE,
        DB_PATH=run_config.PATH.DB_PATH,
        IGNORED_SAMPLES=run_config.PATH.IGNORED_SAMPLES,
        BOX=False,
        SAMPLE_LIST=sample_list,
        DEVICE=device
    )

    # 4. Run on Device
    export_bbox_features(
        KEY_FORMAT=run_config.TYPES.KEY_FORMAT,
        RESIZE=(run_config.SCAN_WIDTH, run_config.SCAN_HEIGHT),
        IMAGE=run_config.PATH.IMAGE,
        DB_PATH=run_config.PATH.DB_PATH,
        IGNORED_SAMPLES=run_config.PATH.IGNORED_SAMPLES,
        BOX=True,
        SAMPLE_LIST=sample_list,
        DEVICE=device
    )


if __name__ == '__main__':
    args = parse_arg()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    if args.prep:
        prep_main(args.exp_type, args.dataset, args.viewpoint, args.box)

    if args.train:
        train_main(args)

    if args.eval:
        eval_main(args)
