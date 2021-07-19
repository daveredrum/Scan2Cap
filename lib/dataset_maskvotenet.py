'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import h5py
import json
import pickle
import random
import numpy as np
import multiprocessing as mp

from itertools import chain
from collections import Counter
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from utils.box_util import get_3d_box
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
OBJ_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
MIN_NUM_OBJ_PTS = 1024

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
SCANREFER_VOCAB = os.path.join(CONF.PATH.DATA, "ScanRefer_vocabulary.json")
SCANREFER_VOCAB_WEIGHTS = os.path.join(CONF.PATH.DATA, "ScanRefer_vocabulary_weights.json")
# MULTIVIEW_DATA = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")

class ReferenceDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_des(self):
        lang = {}
        label = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}
                label[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}
                label[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
                label[scene_id][object_id][ann_id] = {}

            # trim long descriptions
            tokens = data["token"][:CONF.TRAIN.MAX_DES_LEN]

            # tokenize the description
            tokens = ["sos"] + tokens + ["eos"]
            embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2, 300))
            labels = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2)) # start and end

            # load
            for token_id in range(len(tokens)):
                token = tokens[token_id]
                try:
                    embeddings[token_id] = self.glove[token]
                    labels[token_id] = self.vocabulary["word2idx"][token]
                except KeyError:
                    embeddings[token_id] = self.glove["unk"]
                    labels[token_id] = self.vocabulary["word2idx"]["unk"]
            
            # store
            lang[scene_id][object_id][ann_id] = embeddings
            label[scene_id][object_id][ann_id] = labels

        return lang, label

    def _build_vocabulary(self):
        if os.path.exists(SCANREFER_VOCAB):
            self.vocabulary = json.load(open(SCANREFER_VOCAB))
        else:
            if self.split == "train":
                all_words = chain(*[data["token"][:CONF.TRAIN.MAX_DES_LEN] for data in self.scanrefer])
                word_counter = Counter(all_words)
                word_counter = sorted([(k, v) for k, v in word_counter.items() if k in self.glove], key=lambda x: x[1], reverse=True)
                word_list = [k for k, _ in word_counter]

                # build vocabulary
                word2idx, idx2word = {}, {}
                spw = ["pad_", "unk", "sos", "eos"] # NOTE distinguish padding token "pad_" and the actual word "pad"
                for i, w in enumerate(word_list):
                    shifted_i = i + len(spw)
                    word2idx[w] = shifted_i
                    idx2word[shifted_i] = w

                # add special words into vocabulary
                for i, w in enumerate(spw):
                    word2idx[w] = i
                    idx2word[i] = w

                vocab = {
                    "word2idx": word2idx,
                    "idx2word": idx2word
                }
                json.dump(vocab, open(SCANREFER_VOCAB, "w"), indent=4)

                self.vocabulary = vocab

    def _build_frequency(self):
        if os.path.exists(SCANREFER_VOCAB_WEIGHTS):
            with open(SCANREFER_VOCAB_WEIGHTS) as f:
                weights = json.load(f)
                self.weights = np.array([v for _, v in weights.items()])
        else:
            all_tokens = []
            for scene_id in self.lang_ids.keys():
                for object_id in self.lang_ids[scene_id].keys():
                    for ann_id in self.lang_ids[scene_id][object_id].keys():
                        all_tokens += self.lang_ids[scene_id][object_id][ann_id].astype(int).tolist()

            word_count = Counter(all_tokens)
            word_count = sorted([(k, v) for k, v in word_count.items()], key=lambda x: x[0])
            
            # frequencies = [c for _, c in word_count]
            # weights = np.array(frequencies).astype(float)
            # weights = weights / np.sum(weights)
            # weights = 1 / np.log(1.05 + weights)

            weights = np.ones((len(word_count)))

            self.weights = weights
            
            with open(SCANREFER_VOCAB_WEIGHTS, "w") as f:
                weights = {k: v for k, v in enumerate(weights)}
                json.dump(weights, f, indent=4)

    def _load_data(self):
        print("loading data...")
        # load language features
        self.glove = pickle.load(open(GLOVE_PICKLE, "rb"))
        self._build_vocabulary()
        self.num_vocabs = len(self.vocabulary["word2idx"].keys())
        self.lang, self.lang_ids = self._tranform_des()
        self._build_frequency()

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            # self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            # self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy")

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox

class MaskScannetReferenceDataset(ReferenceDataset):
       
    def __init__(self, scanrefer, scanrefer_all_scene, 
        split="train", 
        name="ScanRefer",
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False,
        augment=False,
        debug=False):

        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.name = name
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.debug = debug

        # load data
        self._load_data()
        self.multiview_data = {}

        # fliter
        self.scanrefer = self._filter_object(self.scanrefer)
       
    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17
        ann_id = self.scanrefer[idx]["ann_id"]
        
        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        lang_len = len(self.scanrefer[idx]["token"]) + 2
        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN + 2 else CONF.TRAIN.MAX_DES_LEN + 2

        # get pc data
        target_bboxes = np.zeros((1, 6))
        target_bboxes_mask = np.zeros((1))    
        angle_classes = np.zeros((1,))
        angle_residuals = np.zeros((1,))
        size_classes = np.zeros((1,))
        size_residuals = np.zeros((1, 3))
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        target_bboxes_semcls = np.zeros((1))
        bbox_corner = np.zeros((3, 8))
        
        # load pc data
        (point_cloud, target_bboxes, target_bboxes_mask, 
        angle_classes, angle_residuals, size_classes, size_residuals,
        point_votes, point_votes_mask, target_bboxes_semcls, 
        bbox_corner) = self._get_pc_data(scene_id, object_id)

        data_dict = {}
        # basic info
        data_dict["dataset_idx"] = np.array(idx).astype(np.int64)
        data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)
        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        
        # language data
        data_dict["lang_feat"] = lang_feat.astype(np.float32) # language feature vectors
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict["lang_ids"] = np.array(self.lang_ids[scene_id][str(object_id)][ann_id]).astype(np.int64)

        # point cloud data
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (1, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (1,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (1,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (1,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (1, 3)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (1,) semantic class index
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (1) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["bbox_corner"] = bbox_corner.astype(np.float32) # bounding box corner coordinates
        
        # load time
        data_dict["load_time"] = time.time() - start

        return data_dict

    def _get_pc_data(self, scene_id, object_id):
        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        
        if self.debug:
            point_cloud = point_cloud[:self.num_points]
            instance_labels = instance_labels[:self.num_points]
            semantic_labels = semantic_labels[:self.num_points]
            pcl_color = pcl_color[:self.num_points]
        else:
            # point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
            point_cloud, choices = self._sampling(point_cloud, object_id + 1, instance_labels)
            instance_labels = instance_labels[choices]
            semantic_labels = semantic_labels[choices]
            pcl_color = pcl_color[choices]
        
        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((1, 6))
        target_bboxes_mask = np.zeros((1))    
        angle_classes = np.zeros((1,))
        angle_residuals = np.zeros((1,))
        size_classes = np.zeros((1,))
        size_residuals = np.zeros((1, 3))

        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        
        instance_bboxes_ind = np.where(instance_bboxes[:,-1] == object_id)[0][0]
        target_bboxes_mask[0] = 1
        target_bboxes[0,:] = instance_bboxes[instance_bboxes_ind,0:6]
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment and not self.debug:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
                target_bboxes[:,1] = -1 * target_bboxes[:,1]                                

            # Rotation along X-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotx(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

            # Rotation along Y-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = roty(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

            # Translation
            point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

        # append mask digit
        target_instance_mask = (instance_labels == object_id + 1).astype(np.float32)
        point_cloud = np.concatenate([point_cloud, target_instance_mask[:, np.newaxis]], axis=1)

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        ind = target_instance_mask == 1
        x = point_cloud[ind,:3]
        center = 0.5*(x.min(0) + x.max(0))

        point_votes = center - point_cloud[:, :3]
        point_votes_mask = np.ones(self.num_points)

        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical
        
        class_ind = DC.nyu40id2class[instance_bboxes[instance_bboxes_ind,-2]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0] = class_ind
        size_residuals[0, :] = target_bboxes[0, 3:6] - DC.mean_size_arr[class_ind,:]
            
        target_bboxes_semcls = np.zeros((1))
        target_bboxes_semcls[0] = DC.nyu40id2class[instance_bboxes[instance_bboxes_ind,-2]]

        # construct bounding box corners
        bbox_obb = DC.param2obb(target_bboxes[0, 0:3], 0, 0, size_classes[0], size_residuals[0])
        bbox_corner = get_3d_box(bbox_obb[3:6], bbox_obb[6], bbox_obb[0:3])

        return (point_cloud, target_bboxes, target_bboxes_mask, 
                angle_classes, angle_residuals, size_classes, size_residuals,
                point_votes, point_votes_mask, target_bboxes_semcls, 
                bbox_corner)

    def _filter_object(self, data):
        new_data = []
        cache = []
        for d in data:
            scene_id = d["scene_id"]
            object_id = d["object_id"]

            entry = "{}|{}".format(scene_id, object_id)

            if entry not in cache:
                cache.append(entry)
                new_data.append(d)

        return new_data
    
    def _sampling(self, point_cloud, object_id, instance_labels, min_points=MIN_NUM_OBJ_PTS):
        """
            Sampling N points from the point cloud with respect to the target object
            1) if the target object has less than min_points points, put min_points
                sampled points on the objects with replacement enforcedly and concatenate
                with self.num_points - min_points other randomly sampled points on the 
                whole scene
            2) if the target objects has more than min_points points, sample min_points
                points on the objects without replacement and concatenate
                with self.num_points - min_points other randomly sampled points on the 
                whole scene

            Return:
                point_cloud: (N, num_feat)
                choices: (N)
        """

        ind_mask = instance_labels == object_id

        # sample object points with replacement
        object_point_cloud, object_choices = random_sampling(
            point_cloud[ind_mask], min_points, replace=None, return_choices=True)

        # sample background
        background_point_cloud, background_choices = random_sampling(
            point_cloud, self.num_points - min_points, 
            replace=None, return_choices=True)

        point_cloud = np.concatenate([object_point_cloud, background_point_cloud], axis=0)
        choices = np.concatenate([object_choices, background_choices], axis=0)

        return point_cloud, choices
