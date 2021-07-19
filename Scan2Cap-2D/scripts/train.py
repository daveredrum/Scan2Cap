import os
import json
import argparse
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from lib.dataset import ScanReferDataset
from lib.solver import Solver
from models.snt import ShowAndTell
from models.tdbu import ShowAttendAndTell
from lib.conf import get_config, get_samples, verify_visual_feat


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
    context_size = run_config.PROPOSAL_FEATURE_SIZE
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
            context_size=context_size,
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

    # to CUDA
    model = model.cuda()

    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def get_solver(args, run_config, dataset, dataloader):
    model = get_model(
        args=args,
        run_config=run_config,
        dataset=dataset['train']
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(run_config.PATH.OUTPUT_ROOT, stamp)
        checkpoint = torch.load(os.path.join(run_config.PATH.OUTPUT_ROOT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_" + args.tag.upper()
        root = os.path.join(run_config.PATH.OUTPUT_ROOT, stamp)
        os.makedirs(root, exist_ok=True)

    LR_DECAY_STEP = [1, 2, 4, 8, 10, 12]
    LR_DECAY_RATE = 0.8

    solver = Solver(
        run_config=run_config,
        args=args,
        model=model,
        dataset=dataset,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        criterion='cider'
    )
    num_params = get_num_params(model)

    return solver, num_params, root


def save_info(args, root, num_params, dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(dataset["train"])
    info["num_eval_val"] = len(dataset["eval"]["val"])
    info["num_train_scenes"] = len(dataset["train"].scene_list)
    info["num_eval_val_scenes"] = len(dataset["eval"]["val"].scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def train(args):
    run_config = get_config(
        exp_type=args.exp_type,
        dataset=args.dataset,
        viewpoint=args.viewpoint,
        box=args.box
    )

    train_samples, train_scenes = get_samples(mode='train', key_type=run_config.TYPES.KEY_TYPE)
    val_samples, val_scenes = get_samples(mode='val', key_type=run_config.TYPES.KEY_TYPE)

    train_dset, train_dloader = get_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
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
    dataset = {
        "train": train_dset,
        "eval": {
            "val": val_dset
        }
    }
    dataloader = {
        "train": train_dloader,
        "eval": {
            "val": val_dloader
        }
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, run_config, dataset, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, dataset)
    solver(args.num_epochs, args.verbose)


def train_main(args):
    train(args)
