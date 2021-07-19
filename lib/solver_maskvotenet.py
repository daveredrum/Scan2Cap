'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR


sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from lib.loss_helper_maskvotenet import get_loss
from utils.eta import decode_eta
from lib.pointnet2.pytorch_utils import BNMomentumScheduler


ITER_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[loss] train_objectness_loss: {train_objectness_loss}
[loss] train_vote_loss: {train_vote_loss}
[loss] train_box_loss: {train_box_loss}
[loss] train_sem_cls_loss: {train_sem_cls_loss}
[sco.] train_objn_acc: {train_objn_acc}
[sco.] train_miou: {train_miou}
[sco.] train_sem_acc: {train_sem_acc}
[sco.] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[train] train_objn_acc: {train_objn_acc}
[train] train_miou: {train_miou}
[train] train_sem_acc: {train_sem_acc}
[train] train_pos_ratio: {train_neg_ratio}
[train] train_neg_ratio: {train_neg_ratio}
[val]   val_objn_acc: {val_objn_acc}
[val]   val_miou: {val_miou}
[val]   val_sem_acc: {val_sem_acc}
[val]   val_pos_ratio: {val_neg_ratio}
[val]   val_neg_ratio: {val_neg_ratio}
"""

BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best]  epoch: {epoch}
[best]  miou: {miou}
[best]  sem_acc: {sem_acc}
"""

class Solver():
    def __init__(self, model, config, dataset, dataloader, optimizer, stamp, val_step=10, 
    lr_decay_step=None, lr_decay_rate=None, bn_decay_step=None, bn_decay_rate=None):

        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__
        
        self.model = model
        self.config = config
        self.dataset = dataset
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step

        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.bn_decay_step = bn_decay_step
        self.bn_decay_rate = bn_decay_rate

        self.best = {
            "epoch": 0,
            "miou": -float("inf"),
            "sem_acc": -float("inf")
        }

        # init log
        # contains all necessary info for all phases
        self.log = {
            "train": {},
            "val": {}
        }
        
        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

        # lr scheduler
        if lr_decay_step and lr_decay_rate:
            if isinstance(lr_decay_step, list):
                self.lr_scheduler = MultiStepLR(optimizer, lr_decay_step, lr_decay_rate)
            else:
                self.lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
        else:
            self.lr_scheduler = None

        # bn scheduler
        if bn_decay_step and bn_decay_rate:
            it = -1
            start_epoch = 0
            BN_MOMENTUM_INIT = 0.5
            BN_MOMENTUM_MAX = 0.001
            bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * bn_decay_rate**(int(it / bn_decay_step)), BN_MOMENTUM_MAX)
            self.bn_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)
        else:
            self.bn_scheduler = None

    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * (self._total_iter["train"] / self.val_step)
        
        for epoch_id in range(epoch):
            try:
                self._log("epoch {} starting...".format(epoch_id + 1))

                # feed 
                self._feed(self.dataloader["train"], "train", epoch_id)

                # save model
                self._log("saving last models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

                # update lr scheduler
                if self.lr_scheduler:
                    print("update learning rate --> {}\n".format(self.lr_scheduler.get_last_lr()))
                    self.lr_scheduler.step()

                # update bn scheduler
                if self.bn_scheduler:
                    print("update batch normalization momentum --> {}\n".format(self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))
                    self.bn_scheduler.step()
                
            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _log(self, info_str):
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str)

    def _reset_log(self, phase):
        self.log[phase] = {
            # info
            "forward": [],
            "backward": [],
            "eval": [],
            "fetch": [],
            "iter_time": [],
            # loss (float, not torch.cuda.FloatTensor)
            "loss": [],
            "objectness_loss": [],
            "vote_loss": [],
            "box_loss": [],
            "sem_cls_loss": [],
            # scores (float, not torch.cuda.FloatTensor)
            "miou": [],
            "objn_acc": [],
            "sem_acc": [],
            "neg_ratio": [],
            "pos_ratio": [],
        }

    def _dump_log(self, phase):
        log = {
            "loss": ["loss", "objectness_loss", "vote_loss", "box_loss", "sem_cls_loss"],
            "score": ["miou", "objn_acc", "sem_acc", "pos_ratio", "neg_ratio"]
        }
        for key in log:
            for item in log[key]:
                if self.log[phase][item]:
                    self._log_writer[phase].add_scalar(
                        "{}/{}".format(key, item),
                        np.mean([v for v in self.log[phase][item]]),
                        self._global_iter_id
                    )

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict):
        data_dict = self.model(data_dict)

        return data_dict

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()
        self.optimizer.step()

    def _compute_loss(self, data_dict):
        data_dict = get_loss(
            data_dict=data_dict, 
            config=self.config
        )

        # store loss
        self._running_log["objectness_loss"] = data_dict["objectness_loss"]
        self._running_log["vote_loss"] = data_dict["vote_loss"]
        self._running_log["box_loss"] = data_dict["box_loss"]
        self._running_log["sem_cls_loss"] = data_dict["sem_cls_loss"]
        self._running_log["loss"] = data_dict["loss"]

        # store eval
        self._running_log["miou"] = data_dict["miou"].item()
        self._running_log["objn_acc"] = data_dict["objn_acc"].item()
        self._running_log["sem_acc"] = data_dict["sem_cls_acc"].item()
        self._running_log["pos_ratio"] = data_dict["pos_ratio"].item()
        self._running_log["neg_ratio"] = data_dict["neg_ratio"].item()

    def _record(self, phase):
        # record log
        self.log[phase]["loss"].append(self._running_log["loss"].item())
        self.log[phase]["objectness_loss"].append(self._running_log["objectness_loss"].item())
        self.log[phase]["vote_loss"].append(self._running_log["vote_loss"].item())
        self.log[phase]["box_loss"].append(self._running_log["box_loss"].item())
        self.log[phase]["sem_cls_loss"].append(self._running_log["sem_cls_loss"].item())

        self.log[phase]["sem_acc"].append(self._running_log["sem_acc"])
        self.log[phase]["objn_acc"].append(self._running_log["objn_acc"])
        self.log[phase]["miou"].append(self._running_log["miou"])
        self.log[phase]["pos_ratio"].append(self._running_log["pos_ratio"])
        self.log[phase]["neg_ratio"].append(self._running_log["neg_ratio"])

    def _feed(self, dataloader, phase, epoch_id):
        # switch mode
        self._set_phase(phase)

        # re-init log
        self._reset_log(phase)

        # enter mode
        dataloader = dataloader if phase == "train" else tqdm(dataloader)
        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                data_dict[key] = data_dict[key].cuda()

            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                "objectness_loss": 0,
                "vote_loss": 0,
                "box_loss": 0,
                "sem_cls_loss": 0,
                # acc
                "sem_acc": 0,
                "objn_acc": 0,
                "miou": 0,
                "pos_ratio": 0,
                "neg_ratio": 0,
            }

            # load
            self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())

            if phase == "train":
                # forward
                start = time.time()
                data_dict = self._forward(data_dict)
                self._compute_loss(data_dict)
                self.log[phase]["forward"].append(time.time() - start)

                # backward
                start = time.time()
                self._backward()
                self.log[phase]["backward"].append(time.time() - start)
                
                # eval
                start = time.time()
                # self._eval(data_dict)
                self.log[phase]["eval"].append(time.time() - start)

                # time 
                iter_time = self.log[phase]["fetch"][-1]
                iter_time += self.log[phase]["forward"][-1]
                iter_time += self.log[phase]["backward"][-1]
                iter_time += self.log[phase]["eval"][-1]
                self.log[phase]["iter_time"].append(iter_time)

                # record log
                self._record(phase)

                if (self._global_iter_id + 1) % self.verbose == 0:
                    self._train_report(epoch_id)

                # evaluation
                if self._global_iter_id % self.val_step == 0:
                    # eval on val
                    print("evaluating on val...")
                    self._feed(self.dataloader["val"], "val", epoch_id)
                    self._dump_log("val")


                    self._set_phase("train")
                    self._epoch_report(epoch_id)

                # dump log
                self._dump_log("train")
                self._global_iter_id += 1

            else:
                with torch.no_grad():
                    data_dict = self._forward(data_dict)
                    self._compute_loss(data_dict)

                # record log
                self._record(phase)

        # best
        if phase == "val":
            cur_criterion = "miou"
            cur_best = np.mean(self.log[phase][cur_criterion])
            if cur_best > self.best[cur_criterion]:
                self._log("best {} achieved: {}".format(cur_criterion, cur_best))

                self.best["epoch"] = epoch_id + 1
                self.best["miou"] = np.mean(self.log[phase]["miou"])
                self.best["sem_acc"] = np.mean(self.log[phase]["sem_acc"])
                
                # save model
                self._log("saving best models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))

    def _finish(self, epoch_id):
        # print best
        self._best_report()

        # save check point
        self._log("saving checkpoint...\n")
        save_dict = {
            "epoch": epoch_id,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint.tar"))

        # save model
        self._log("saving last models...\n")
        model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

        # export
        for phase in ["train", "val"]:
            self._log_writer[phase].export_scalars_to_json(os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        eval_time = self.log["train"]["eval"]
        iter_time = self.log["train"]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        
        num_train_iter_left = self._total_iter["train"] - self._global_iter_id - 1
        eta_sec = num_train_iter_left * mean_train_time
        
        num_val_times = num_train_iter_left // self.val_step
        eta_sec += len(self.dataloader["val"]) * num_val_times * mean_est_val_time
        
        eta = decode_eta(eta_sec)

        # print report
        iter_report = self.__iter_report_template.format(
            epoch_id=epoch_id + 1,
            iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["train"],
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
            train_vote_loss=round(np.mean([v for v in self.log["train"]["vote_loss"]]), 5),
            train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
            train_sem_cls_loss=round(np.mean([v for v in self.log["train"]["sem_cls_loss"]]), 5),
            train_objn_acc=round(np.mean([v for v in self.log["train"]["objn_acc"]]), 5),
            train_miou=round(np.mean([v for v in self.log["train"]["miou"]]), 5),
            train_sem_acc=round(np.mean([v for v in self.log["train"]["sem_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
            mean_fetch_time=round(np.mean(fetch_time), 5),
            mean_forward_time=round(np.mean(forward_time), 5),
            mean_backward_time=round(np.mean(backward_time), 5),
            mean_eval_time=round(np.mean(eval_time), 5),
            mean_iter_time=round(np.mean(iter_time), 5),
            eta_h=eta["h"],
            eta_m=eta["m"],
            eta_s=eta["s"]
        )
        self._log(iter_report)

    def _epoch_report(self, epoch_id):
        self._log("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
        epoch_report = self.__epoch_report_template.format(
            train_objn_acc=round(np.mean([v for v in self.log["train"]["objn_acc"]]), 5),
            train_miou=round(np.mean([v for v in self.log["train"]["miou"]]), 5),
            train_sem_acc=round(np.mean([v for v in self.log["train"]["sem_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
            val_objn_acc=round(np.mean([v for v in self.log["val"]["objn_acc"]]), 5),
            val_miou=round(np.mean([v for v in self.log["val"]["miou"]]), 5),
            val_sem_acc=round(np.mean([v for v in self.log["val"]["sem_acc"]]), 5),
            val_pos_ratio=round(np.mean([v for v in self.log["val"]["pos_ratio"]]), 5),
            val_neg_ratio=round(np.mean([v for v in self.log["val"]["neg_ratio"]]), 5),
        )
        self._log(epoch_report)
    
    def _best_report(self):
        self._log("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            miou=round(self.best["miou"], 5),
            sem_acc=round(self.best["sem_acc"], 5)
        )
        self._log(best_report)
        with open(os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)
