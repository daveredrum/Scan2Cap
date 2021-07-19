import os
import argparse
from collections import OrderedDict
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.dataset import ScanReferDataset
from models.snt import ShowAndTell
from models.tdbu import ShowAttendAndTell
from models.retr import Retrieval2D
from lib.conf import get_config, get_samples, verify_visual_feat
from lib.eval_helper import eval_cap
import h5py

def get_dataloader(batch_size, num_workers, shuffle, sample_list, scene_list, run_config, split):
    dataset = ScanReferDataset(
        split=split,
        sample_list=sample_list,
        scene_list=scene_list,
        run_config=run_config
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            collate_fn=dataset.collate_fn)

    return dataset, dataloader


def get_model(args, run_config, dataset):
    model_selection = args.model
    feat_size = 0
    add_global, add_target, add_context = verify_visual_feat(args.visual_feat)

    if add_global:
        feat_size += run_config.GLOBAL_FEATURE_SIZE
    if add_target:
        feat_size += run_config.TARGET_FEATURE_SIZE

    assert feat_size != 0

    if add_context and model_selection == 'satnt':
        print("Using Show, Attend and Tell.")
        model = ShowAttendAndTell(
            device='cuda',
            max_desc_len=run_config.MAX_DESC_LEN,
            vocabulary=dataset.vocabulary,
            embeddings=dataset.glove,
            emb_size=run_config.EMBEDDING_SIZE,
            feat_size=feat_size,
            context_size=run_config.PROPOSAL_FEATURE_SIZE,
            feat_input={'add_global': add_global, 'add_target': add_target},
            hidden_size=run_config.DECODER_HIDDEN_SIZE,
        )

    elif model_selection == 'snt' and not add_context:
        model = ShowAndTell(
            device='cuda',
            max_desc_len=run_config.MAX_DESC_LEN,
            vocabulary=dataset.vocabulary,
            embeddings=dataset.glove,
            emb_size=run_config.EMBEDDING_SIZE,
            feat_size=feat_size,
            feat_input={'add_global': add_global, 'add_target': add_target},
            hidden_size=run_config.DECODER_HIDDEN_SIZE,
        )

    else:
        raise NotImplementedError('Requested model {} is not implemented.'.format(dataset))

    # Load checkpoint
    if args.ckpt_path is not None:
        checkpoint = torch.load(args.ckpt_path)
        # print(checkpoint.keys())
        try:
            model.load_state_dict(checkpoint, strict=True)
            print("Loaded checkpoint from {}".format(args.ckpt_path))
        except KeyError:
            print("Checkpoint has the following keys available: ")
            print(checkpoint.keys())
            exit(0)
    else:
        print("No checkpoint specified. Please specify one by --ckpt_path.")
        exit(0)

    # to CUDA
    model = model.cuda()

    return model


def get_retrieval_model(args, run_config, train_dataset):
    _ = args

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.tag: stamp += "_" + args.tag.upper()
    retrieval_directory = os.path.join(run_config.PATH.OUTPUT_ROOT, stamp)
    os.makedirs(retrieval_directory, exist_ok=True)

    feat_size = run_config.TARGET_FEATURE_SIZE - 4
    train_scene_list = train_dataset.scene_list
    scanrefer = h5py.File(run_config.PATH.DB_PATH, 'r')
    scanrefer_box_features = scanrefer['boxfeat']
    scanrefer_oids = scanrefer['objectids']
    # only take features that are in the train set and describe the target object.
    ordered_train_feature_matrix = []

    for sample_id, v in scanrefer_box_features.items():
        if sample_id.split('-')[0] in train_scene_list:
            target_object_id = int(sample_id.split('-')[1].split('_')[0])
            object_ids = np.array(scanrefer_oids[sample_id])
            target_idx = np.where(object_ids == int(target_object_id))[0]
            object_feature = np.array(v)[target_idx, :].reshape(-1, feat_size)
            ordered_train_feature_matrix.append((sample_id, object_feature))

    ordered_train_feature_matrix = OrderedDict(ordered_train_feature_matrix)

    model = Retrieval2D(
        db_path=os.path.join(retrieval_directory, 'train_memory_map.dat'),
        feat_size=feat_size,
        vis_feat_dict=ordered_train_feature_matrix,
        lang_ids=train_dataset.lang_ids
    )

    model.cuda()
    model.eval()

    return model, retrieval_directory


def eval_caption(args):
    run_config = get_config(
        exp_type=args.exp_type,
        dataset=args.dataset,
        viewpoint=args.viewpoint,
        box=args.box
    )
    train_samples, train_scenes = get_samples(mode='train', key_type=run_config.TYPES.KEY_TYPE)
    val_samples, val_scenes = get_samples(mode='val', key_type=run_config.TYPES.KEY_TYPE)
    
    print('Number of training samples: ', len(train_samples))
    print('Number of validation samples: ', len(val_samples))
    
    train_dset, train_dloader = get_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        sample_list=train_samples,
        scene_list=train_scenes,
        run_config=run_config,
        split='train'
    )

    val_dset, val_dloader = get_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        sample_list=val_samples,
        scene_list=val_scenes,
        run_config=run_config,
        split='val'
    )

    retr_dir = None
    folder = args.folder
    if args.exp_type == 'ret':
        model, retr_dir = get_retrieval_model(args=args, run_config=run_config, train_dataset=train_dset)
    elif args.exp_type == 'nret':
        model = get_model(args=args, run_config=run_config, dataset=val_dset)
    else:
        raise NotImplementedError('exp_type {} is not implemented.'.format(args.exp_type))

    # evaluate
    if retr_dir is not None:
        folder = retr_dir
    
    assert folder is not None
    bleu, cider, rouge, meteor = eval_cap(
        _global_iter_id=0,
        model=model,
        dataset=val_dset,
        dataloader=val_dloader,
        phase='val',
        folder=folder,
        max_len=run_config.MAX_DESC_LEN,
        mode=args.exp_type,
        extras=args.extras,
        is_eval=True
    )

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

def eval_main(args):
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    eval_caption(args)
