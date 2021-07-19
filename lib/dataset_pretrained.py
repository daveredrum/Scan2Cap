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
from utils.box_util import get_3d_box, get_3d_box_batch
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
OBJ_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
MIN_NUM_OBJ_PTS = 1024
NUM_PRESET_EPOCHS = 100
# NUM_PRESET_EPOCHS = 1 # DEBUG
NUM_PROPOSALS = 256

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

class PretrainedGTDataset(ReferenceDataset):
       
    # def __init__(self, scanrefer, scanrefer_all_scene, 
    #     split="train", 
    #     name="ScanRefer",
    #     augment=False,
    #     debug=False,
    #     scan2cad_rotation=None):

    #     self.scanrefer = scanrefer
    #     self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
    #     self.split = split
    #     self.name = name
    #     self.augment = augment
    #     self.debug = debug
    #     self.scan2cad_rotation = scan2cad_rotation

    #     # load data
    #     self._load_data()
    #     self.gt_feature_data = {}

    #     # fliter
    #     self.scene_objects = self._get_scene_objects(self.scanrefer)
       
    # def __len__(self):
    #     return len(self.scanrefer)

    # def __getitem__(self, idx):
    #     start = time.time()
    #     scene_id = self.scanrefer[idx]["scene_id"]
    #     object_id = int(self.scanrefer[idx]["object_id"])
    #     object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
    #     object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17
    #     ann_id = self.scanrefer[idx]["ann_id"]
        
    #     # get language features
    #     lang_feat = self.lang[scene_id][str(object_id)][ann_id]
    #     lang_len = len(self.scanrefer[idx]["token"]) + 2
    #     lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN + 2 else CONF.TRAIN.MAX_DES_LEN + 2

    #     # get pc data
    #     bbox_object_ids = np.zeros((MAX_NUM_OBJ,))
    #     bbox_feature = np.zeros((MAX_NUM_OBJ, 128))
    #     bbox_corner = np.zeros((MAX_NUM_OBJ, 8, 3))
    #     bbox_center = np.zeros((MAX_NUM_OBJ, 3))
    #     bbox_mask = np.zeros((MAX_NUM_OBJ,))

    #     # load all bbox data in the scene
    #     gt_object_ids, gt_features, gt_corners, gt_centers = self._get_feature(scene_id)
    #     num_valid_objects = gt_object_ids.shape[0]
    #     bbox_object_ids[:num_valid_objects] = gt_object_ids
    #     bbox_feature[:num_valid_objects] = gt_features
    #     bbox_corner[:num_valid_objects] = gt_corners
    #     bbox_center[:num_valid_objects] = gt_centers
    #     bbox_mask[:num_valid_objects] = 1

    #     bbox_idx = 0
    #     for i in range(len(gt_object_ids)):
    #         if gt_object_ids[i] == object_id:
    #             bbox_idx = i

    #     # object rotations
    #     scene_object_rotations = np.zeros((MAX_NUM_OBJ, 3, 3))
    #     scene_object_rotation_masks = np.zeros((MAX_NUM_OBJ,)) # NOTE this is not object mask!!!
    #     # if scene is not in scan2cad annotations, skip
    #     # if the instance is not in scan2cad annotations, skip
    #     if self.scan2cad_rotation and scene_id in self.scan2cad_rotation:
    #         for i, instance_id in enumerate(gt_object_ids):
    #             try:
    #                 rotation = np.array(self.scan2cad_rotation[scene_id][str(instance_id)])

    #                 scene_object_rotations[i] = rotation
    #                 scene_object_rotation_masks[i] = 1
    #             except KeyError:
    #                 pass

    #     data_dict = {}
    #     # basic info
    #     data_dict["dataset_idx"] = np.array(idx).astype(np.int64)
    #     data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)
    #     data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
    #     data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        
    #     # language data
    #     data_dict["lang_feat"] = lang_feat.astype(np.float32) # language feature vectors
    #     data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
    #     data_dict["lang_ids"] = np.array(self.lang_ids[scene_id][str(object_id)][ann_id]).astype(np.int64)

    #     # point cloud data
    #     data_dict["bbox_object_ids"] = bbox_object_ids.astype(np.int32) # point cloud features
    #     data_dict["bbox_feature"] = bbox_feature.astype(np.float32) # point cloud features
    #     data_dict["bbox_corner"] = bbox_corner.astype(np.float32) # bounding box corner coordinates
    #     data_dict["bbox_center"] = bbox_center.astype(np.float32) # bounding box corner coordinates
    #     data_dict["bbox_mask"] = bbox_mask # mask indicating the valid objects
    #     data_dict["bbox_idx"] = bbox_idx # idx for the target object

    #     # ground truth data
    #     data_dict["bbox_corner_label"] = bbox_corner.astype(np.float64) # target box corners NOTE type must be double
    #     data_dict["bbox_center_label"] = bbox_center.astype(np.float64) # target box corners NOTE type must be double

    #     # rotation data
    #     data_dict["scene_object_rotations"] = scene_object_rotations.astype(np.float32) # (MAX_NUM_OBJ, 3, 3)
    #     data_dict["scene_object_rotation_masks"] = scene_object_rotation_masks.astype(np.int64) # (MAX_NUM_OBJ)
        
    #     # load time
    #     data_dict["load_time"] = time.time() - start

    #     return data_dict

    def __init__(self, scanrefer, scanrefer_all_scene, 
        split="train", 
        name="ScanRefer",
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False,
        debug=False,
        scan2cad_rotation=None):

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
        self.scan2cad_rotation = scan2cad_rotation

        # load data
        # self._load_data(name)
        self._load_data()
        self.multiview_data = {}
        self.gt_feature_data = {}

        # fliter
        self.scene_objects = self._get_scene_objects(self.scanrefer)
       
    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = self.scanrefer[idx]["ann_id"]

        annotated = 1

        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17
        
        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        lang_len = len(self.scanrefer[idx]["token"]) + 2
        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN + 2 else CONF.TRAIN.MAX_DES_LEN + 2

        lang_ids = self.lang_ids[scene_id][str(object_id)][ann_id]

        unique_multiple_flag = self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3] # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6] 
            point_cloud[:, 3:6] = (point_cloud[:, 3:6]-MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:, 6:9]
            point_cloud = np.concatenate([point_cloud, normals], 1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview], 1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        ref_box_label = np.zeros(MAX_NUM_OBJ) # bbox label for reference target
        ref_center_label = np.zeros(3) # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3) # bbox size residual for reference target
        ref_box_corner_label = np.zeros((8, 3))

        num_bbox = 1
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        
        num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
        target_bboxes_mask[0:num_bbox] = 1
        target_bboxes[0:num_bbox, :] = instance_bboxes[:MAX_NUM_OBJ, 0:6]
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment:
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

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label            
            if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind, :3]
                center = 0.5*(x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
        
        class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox, -2]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:num_bbox] = class_ind
        size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]

        # construct the reference target label for each bbox
        if object_id != -1:
            for i, gt_id in enumerate(instance_bboxes[:num_bbox,-1]):
                if gt_id == object_id:
                    ref_box_label[i] = 1
                    ref_center_label = target_bboxes[i, 0:3]
                    ref_heading_class_label = angle_classes[i]
                    ref_heading_residual_label = angle_residuals[i]
                    ref_size_class_label = size_classes[i]
                    ref_size_residual_label = size_residuals[i]

                    # construct ground truth box corner coordinates
                    ref_obb = DC.param2obb(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                    ref_size_class_label, ref_size_residual_label)
                    ref_box_corner_label = get_3d_box(ref_obb[3:6], ref_obb[6], ref_obb[0:3])

        # construct all GT bbox corners
        all_obb = DC.param2obb_batch(target_bboxes[:num_bbox, 0:3], angle_classes[:num_bbox].astype(np.int64), angle_residuals[:num_bbox],
                                size_classes[:num_bbox].astype(np.int64), size_residuals[:num_bbox])
        all_box_corner_label = get_3d_box_batch(all_obb[:, 3:6], all_obb[:, 6], all_obb[:, 0:3])
        
        # store
        gt_box_corner_label = np.zeros((MAX_NUM_OBJ, 8, 3))
        gt_box_masks = np.zeros((MAX_NUM_OBJ,))
        gt_box_object_ids = np.zeros((MAX_NUM_OBJ,))

        gt_box_corner_label[:num_bbox] = all_box_corner_label
        gt_box_masks[:num_bbox] = 1
        gt_box_object_ids[:num_bbox] = instance_bboxes[:, -1]

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_object_ids = np.zeros((MAX_NUM_OBJ,)) # object ids of all objects
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
            target_object_ids[0:num_bbox] = instance_bboxes[:, -1][0:num_bbox]
        except KeyError:
            pass

        # get bbox
        bbox_object_ids = np.zeros((MAX_NUM_OBJ,))
        bbox_feature = np.zeros((MAX_NUM_OBJ, 128))
        bbox_corner = np.zeros((MAX_NUM_OBJ, 8, 3))
        bbox_center = np.zeros((MAX_NUM_OBJ, 3))
        bbox_mask = np.zeros((MAX_NUM_OBJ,))

        # load all bbox data in the scene
        gt_object_ids, gt_features, gt_corners, gt_centers = self._get_feature(scene_id)
        num_valid_objects = gt_object_ids.shape[0]
        bbox_object_ids[:num_valid_objects] = gt_object_ids
        bbox_feature[:num_valid_objects] = gt_features
        bbox_corner[:num_valid_objects] = gt_corners
        bbox_center[:num_valid_objects] = gt_centers
        bbox_mask[:num_valid_objects] = 1

        bbox_idx = 0
        for i in range(len(gt_object_ids)):
            if gt_object_ids[i] == object_id:
                bbox_idx = i

        # object rotations
        scene_object_rotations = np.zeros((MAX_NUM_OBJ, 3, 3))
        scene_object_rotation_masks = np.zeros((MAX_NUM_OBJ,)) # NOTE this is not object mask!!!
        # if scene is not in scan2cad annotations, skip
        # if the instance is not in scan2cad annotations, skip
        if self.scan2cad_rotation and scene_id in self.scan2cad_rotation:
            for i, instance_id in enumerate(gt_object_ids):
                try:
                    rotation = np.array(self.scan2cad_rotation[scene_id][str(instance_id)])

                    scene_object_rotations[i] = rotation
                    scene_object_rotation_masks[i] = 1
                except KeyError:
                    pass

        data_dict = {}
        # pc
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict["pcl_color"] = pcl_color

        # basic info
        data_dict["dataset_idx"] = np.array(idx).astype(np.int64)
        data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)
        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        data_dict["annotated"] = np.array(annotated).astype(np.int64)
        
        # language data
        data_dict["lang_feat"] = lang_feat.astype(np.float32) # language feature vectors
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict["lang_ids"] = np.array(self.lang_ids[scene_id][str(object_id)][ann_id]).astype(np.int64)

        # GT bbox data
        data_dict["bbox_object_ids"] = bbox_object_ids.astype(np.int32) # point cloud features
        data_dict["bbox_feature"] = bbox_feature.astype(np.float32) # point cloud features
        data_dict["bbox_corner"] = bbox_corner.astype(np.float32) # bounding box corner coordinates
        data_dict["bbox_center"] = bbox_center.astype(np.float32) # bounding box corner coordinates
        data_dict["bbox_mask"] = bbox_mask.astype(np.int64) # mask indicating the valid objects
        data_dict["bbox_idx"] = bbox_idx # idx for the target object

        # object detection labels
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index
        data_dict["scene_object_ids"] = target_object_ids.astype(np.int64) # (MAX_NUM_OBJ,) object ids of all objects
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)

        # localization labels
        data_dict["ref_box_label"] = ref_box_label.astype(np.int64) # 0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)
        data_dict["ref_heading_class_label"] = np.array(int(ref_heading_class_label)).astype(np.int64)
        data_dict["ref_heading_residual_label"] = np.array(int(ref_heading_residual_label)).astype(np.int64)
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(np.int64)
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)
        data_dict["ref_box_corner_label"] = ref_box_corner_label.astype(np.float64) # target box corners NOTE type must be double
        data_dict["unique_multiple"] = np.array(unique_multiple_flag).astype(np.int64)

        # ground truth data
        data_dict["bbox_corner_label"] = bbox_corner.astype(np.float64) # target box corners NOTE type must be double
        data_dict["bbox_center_label"] = bbox_center.astype(np.float64) # target box corners NOTE type must be double

        data_dict["gt_box_corner_label"] = gt_box_corner_label.astype(np.float64) # all GT box corners NOTE type must be double
        data_dict["gt_box_masks"] = gt_box_masks.astype(np.int64) # valid bbox masks
        data_dict["gt_box_object_ids"] = gt_box_object_ids.astype(np.int64) # valid bbox object ids

        # rotation data
        data_dict["scene_object_rotations"] = scene_object_rotations.astype(np.float32) # (MAX_NUM_OBJ, 3, 3)
        data_dict["scene_object_rotation_masks"] = scene_object_rotation_masks.astype(np.int64) # (MAX_NUM_OBJ)
        
        # misc
        data_dict["load_time"] = time.time() - start

        return data_dict

    def _nn_distance(self, pc1, pc2):
        N = pc1.shape[0]
        M = pc2.shape[0]
        pc1_expand_tile = pc1[:, np.newaxis]
        pc2_expand_tile = pc2[np.newaxis, :]
        pc_diff = pc1_expand_tile - pc2_expand_tile

        pc_dist = np.sum(pc_diff**2, axis=-1) # (N,M)
        idx1 = np.argmin(pc_dist, axis=1) # (N)
        idx2 = np.argmin(pc_dist, axis=0) # (M)

        return idx1, idx2

    def _get_bbox_centers(self, corners):
        coord_min = np.min(corners, axis=1) # num_bboxes, 3
        coord_max = np.max(corners, axis=1) # num_bboxes, 3

        return (coord_min + coord_max) / 2

    def _get_feature(self, scene_id, num_epochs=NUM_PRESET_EPOCHS):
        pid = mp.current_process().pid
        if pid not in self.gt_feature_data:
            # database_path = os.path.join(CONF.PATH.VOTENET_FEATURES.format(self.name), "{}.hdf5".format(self.split))
            database_path = os.path.join(CONF.PATH.GT_FEATURES.format(self.name), "{}.hdf5".format(self.split))
            self.gt_feature_data[pid] = h5py.File(database_path, "r", libver="latest")

        # pick out the features for the train split for a random epoch
        # the epoch pointer is always 0 in the eval mode for train split
        # this doesn't apply to val split
        if self.split == "train" and self.augment and not self.debug:
            epoch_id = random.choice(range(num_epochs))
        else:
            epoch_id = 0

        # load object bounding box information
        gt_object_ids = self.gt_feature_data[pid]["{}|{}_gt_ids".format(epoch_id, scene_id)]

        # # load object bounding box information
        # pred_corners = self.gt_feature_data[pid]["{}|{}_bbox_corners".format(epoch_id, scene_id)]
        # pred_centers = self._get_bbox_centers(pred_corners)

        gt_corners = self.gt_feature_data[pid]["{}|{}_gt_corners".format(epoch_id, scene_id)]
        gt_centers = self._get_bbox_centers(gt_corners)

        # load object features
        # _, assignments = self._nn_distance(pred_centers, gt_centers)
        gt_features = self.gt_feature_data[pid]["{}|{}_features".format(epoch_id, scene_id)]
        # gt_features = np.take(gt_features, assignments.astype(np.int64), axis=0)

        return np.array(gt_object_ids), np.array(gt_features), np.array(gt_corners), np.array(gt_centers)

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

    def _get_scene_objects(self, data):
        scene_objects = {}
        for d in data:
            scene_id = d["scene_id"]
            object_id = d["object_id"]

            if scene_id not in scene_objects:
                scene_objects[scene_id] = []

            if object_id not in scene_objects[scene_id]:
                scene_objects[scene_id].append(object_id)

        return scene_objects

class PretrainedVoteNetDataset(ReferenceDataset):
       
    def __init__(self, scanrefer, scanrefer_all_scene, 
        split="train", 
        name="ScanRefer",
        augment=False,
        debug=False,
        scan2cad_rotation=None):

        # NOTE only feed the scan2cad_rotation when on the training mode and train split

        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.name = name
        self.augment = augment
        self.debug = debug
        self.scan2cad_rotation = scan2cad_rotation

        # load data
        self._load_data()
        self.votenet_feature_data = {}
        self.object_id_to_sem_cls = self._get_sem_label_mapping()

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
        bbox_object_ids = np.zeros((NUM_PROPOSALS,))
        bbox_sem_cls = np.zeros((NUM_PROPOSALS,))
        bbox_feature = np.zeros((NUM_PROPOSALS, 128))
        bbox_corner = np.zeros((NUM_PROPOSALS, 8, 3))
        bbox_center = np.zeros((NUM_PROPOSALS, 3))
        bbox_mask = np.zeros((NUM_PROPOSALS,))
        bbox_objectness = np.zeros((NUM_PROPOSALS, 2))
        bbox_sem_cls_scores = np.zeros((NUM_PROPOSALS, 18))

        bbox_object_ids_label = np.zeros((MAX_NUM_OBJ,))
        bbox_sem_cls_label = np.zeros((NUM_PROPOSALS,))
        bbox_corner_label = np.zeros((MAX_NUM_OBJ, 8, 3))
        bbox_center_label = np.zeros((MAX_NUM_OBJ, 3))
        bbox_mask_label = np.zeros((MAX_NUM_OBJ,))

        ref_box_corner_label = np.zeros((MAX_NUM_OBJ, 8, 3))
        ref_box_center_label = np.zeros((MAX_NUM_OBJ, 3))

        # load pretrained VoteNet bbox data in the scene
        (votenet_object_ids, votenet_features, votenet_bbox_corners, votenet_bbox_centers,
                gt_object_ids, gt_bbox_corners, gt_bbox_centers) = self._get_feature(scene_id)

        num_valid_boxes = votenet_object_ids.shape[0]
        bbox_object_ids[:num_valid_boxes] = votenet_object_ids
        bbox_feature[:num_valid_boxes] = votenet_features
        bbox_corner[:num_valid_boxes] = votenet_bbox_corners
        bbox_center[:num_valid_boxes] = votenet_bbox_centers
        bbox_mask[:num_valid_boxes] = 1

        num_gt_boxes = gt_object_ids.shape[0]
        bbox_object_ids_label[:num_gt_boxes] = gt_object_ids
        bbox_corner_label[:num_gt_boxes] = gt_bbox_corners
        bbox_center_label[:num_gt_boxes] = gt_bbox_centers

        # construct fake objectness scores
        bbox_objectness.fill(1e-8)
        bbox_objectness[:num_valid_boxes, 1] = 1 - 1e-8
        
        bbox_sem_cls_scores.fill(1e-8)
        for i in range(num_valid_boxes):
            sem_cls = self.object_id_to_sem_cls[scene_id][bbox_object_ids[i]]
            
            # store
            bbox_sem_cls[i] = sem_cls
            bbox_sem_cls_scores[i, sem_cls] = 1 - 1e-8

        if self.augment: # use augment flag to indicate train mode
            bbox_object_ids_label = np.array([object_id])
            ind = np.where(gt_object_ids == object_id)[0][0]
            ref_box_corner_label = gt_bbox_corners[ind]
            ref_box_center_label = gt_bbox_centers[ind]
            bbox_mask_label = np.array([1])

            # convert object ids to semantic labels
            bbox_sem_cls_label = np.array([self.object_id_to_sem_cls[scene_id][bbox_object_ids_label[0]]])
        else:
            num_valid_boxes = gt_object_ids.shape[0]
            bbox_object_ids_label[:num_valid_boxes] = gt_object_ids
            ref_box_corner_label[:num_valid_boxes] = gt_bbox_corners
            ref_box_center_label[:num_valid_boxes] = gt_bbox_centers
            bbox_mask_label[:num_valid_boxes] = 1

            # convert object ids to semantic labels
            for i in range(num_valid_boxes):
                bbox_sem_cls_label[i] = self.object_id_to_sem_cls[scene_id][bbox_object_ids_label[i]]

        # object rotations
        scene_object_rotations = np.zeros((MAX_NUM_OBJ, 3, 3))
        scene_object_rotation_masks = np.zeros((MAX_NUM_OBJ,)) # NOTE this is not object mask!!!
        # if scene is not in scan2cad annotations, skip
        # if the instance is not in scan2cad annotations, skip
        if self.scan2cad_rotation and scene_id in self.scan2cad_rotation:
            for i, instance_id in enumerate(bbox_object_ids_label[:num_gt_boxes].astype(int)):
                try:
                    rotation = np.array(self.scan2cad_rotation[scene_id][str(instance_id)])

                    scene_object_rotations[i] = rotation
                    scene_object_rotation_masks[i] = 1
                except KeyError:
                    pass

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
        data_dict["bbox_feature"] = bbox_feature.astype(np.float32) # point cloud features
        data_dict["bbox_corner"] = bbox_corner.astype(np.float64) # bounding box corner coordinates NOTE type must be double
        data_dict["bbox_center"] = bbox_center.astype(np.float64) # bounding box center coordinates NOTE type must be double
        data_dict["bbox_mask"] = bbox_mask # mask indicating the valid objects
        data_dict["bbox_object_ids"] = bbox_object_ids.astype(np.int64) # (MAX_NUM_OBJ,) object ids of all objects
        data_dict["sem_cls"] = bbox_sem_cls.astype(np.int64) # (MAX_NUM_OBJ,) semantic labels of all objects
        data_dict["sem_cls_scores"] = bbox_sem_cls_scores
        data_dict["objectness_scores"] = bbox_objectness

        # ground truth data
        data_dict["bbox_corner_label"] = bbox_corner_label.astype(np.float64) # target box corners NOTE type must be double
        data_dict["bbox_center_label"] = bbox_center_label.astype(np.float64) # target box corners NOTE type must be double
        data_dict["scene_object_ids"] = bbox_object_ids_label.astype(np.int64) # (MAX_NUM_OBJ,) object ids of all objects
        data_dict["sem_cls_label"] = bbox_sem_cls_label.astype(np.int64) # (MAX_NUM_OBJ,) semantic labels of all objects

        # ground truth target data
        data_dict["ref_box_corner_label"] = ref_box_corner_label.astype(np.float64) # target box corners NOTE type must be double
        data_dict["ref_box_center_label"] = ref_box_center_label.astype(np.float64) # target box corners NOTE type must be double
        data_dict["ref_box_mask"] = bbox_mask_label # mask indicating the valid objects

        # rotation data
        data_dict["scene_object_rotations"] = scene_object_rotations.astype(np.float32) # (MAX_NUM_OBJ, 3, 3)
        data_dict["scene_object_rotation_masks"] = scene_object_rotation_masks.astype(np.int64) # (MAX_NUM_OBJ)

        # load time
        data_dict["load_time"] = time.time() - start

        return data_dict

    def _get_bbox_centers(self, corners):
        coord_min = np.min(corners, axis=1) # num_bboxes, 3
        coord_max = np.max(corners, axis=1) # num_bboxes, 3

        return (coord_min + coord_max) / 2

    def _get_feature(self, scene_id, num_epochs=NUM_PRESET_EPOCHS):
        pid = mp.current_process().pid
        if pid not in self.votenet_feature_data:
            # votenet
            database_path = os.path.join(CONF.PATH.VOTENET_FEATURES.format(self.name), "{}.hdf5".format(self.split))
            self.votenet_feature_data[pid] = h5py.File(database_path, "r", libver="latest")

        # pick out the features for the train split for a random epoch
        # the epoch pointer is always 0 in the eval mode for train split
        # this doesn't apply to val split
        if self.split == "train" and self.augment and not self.debug:
            epoch_id = random.choice(range(num_epochs))
        else:
            epoch_id = 0

        # load object ids
        votenet_object_ids = self.votenet_feature_data[pid]["{}|{}_object_ids".format(epoch_id, scene_id)]
        votenet_object_ids = np.array(votenet_object_ids)

        # load object features
        votenet_features = self.votenet_feature_data[pid]["{}|{}_features".format(epoch_id, scene_id)]
        votenet_features = np.array(votenet_features)

        # load object bounding box information
        votenet_bbox_corners = self.votenet_feature_data[pid]["{}|{}_bbox_corners".format(epoch_id, scene_id)]
        votenet_bbox_corners = np.array(votenet_bbox_corners)

        # construct box centers
        votenet_bbox_centers = self._get_bbox_centers(votenet_bbox_corners)

        # load GT object ids
        gt_object_ids = self.votenet_feature_data[pid]["{}|{}_gt_ids".format(epoch_id, scene_id)]
        gt_object_ids = np.array(gt_object_ids)

        # load GT object bounding box information
        gt_bbox_corners = self.votenet_feature_data[pid]["{}|{}_gt_corners".format(epoch_id, scene_id)]
        gt_bbox_corners = np.array(gt_bbox_corners)

        # construct GT box centers
        gt_bbox_centers = self._get_bbox_centers(gt_bbox_corners)

        # ----------- DEBUG -----------
        # from scripts.visualize_pretrained_bbox import write_bbox, align_mesh
        # from utils.pc_utils import write_ply
        # from tqdm import tqdm

        # dump_dir = os.path.join(CONF.PATH.BASE, "pretrained_bbox", scene_id)
        # os.makedirs(dump_dir, exist_ok=True)

        # print("visualizing the scene...")
        # mesh = align_mesh(scene_id)
        # mesh.write(os.path.join(dump_dir, 'mesh.ply'))
        
        # pc = self.votenet_feature_data[pid]["{}|{}_pc".format(epoch_id, scene_id)]
        # pc = np.array(pc)
        # write_ply(pc[:, :3], os.path.join(dump_dir, 'pc.ply'))

        # print("visualizing all pretrained bboxes...")
        # bbox_corners = votenet_bbox_corners
        # bbox_object_ids = votenet_object_ids
        # gt_dump_dir = os.path.join(dump_dir, "votenet")
        # os.makedirs(gt_dump_dir, exist_ok=True)
        # for i in tqdm(range(bbox_object_ids.shape[0])):
        #     corner = bbox_corners[i]
        #     object_id = bbox_object_ids[i]
        #     write_bbox(corner, 1, os.path.join(gt_dump_dir, "{}_votenet.ply".format(str(object_id))))

        # print("visualizing all GT bboxes...")
        # bbox_corners = gt_bbox_corners
        # bbox_object_ids = gt_object_ids
        # gt_dump_dir = os.path.join(dump_dir, "gt")
        # os.makedirs(gt_dump_dir, exist_ok=True)
        # for i in tqdm(range(bbox_object_ids.shape[0])):
        #     corner = bbox_corners[i]
        #     object_id = bbox_object_ids[i]
        #     write_bbox(corner, 0, os.path.join(gt_dump_dir, "{}_gt.ply".format(str(object_id))))

        # print("done!")
        # exit()

        return votenet_object_ids, votenet_features, votenet_bbox_corners, votenet_bbox_centers, \
                gt_object_ids, gt_bbox_corners, gt_bbox_centers

    def _get_sem_label_mapping(self):
        object_id_to_sem_cls = {}
        for scene_id in self.scene_list:
            object_id_to_sem_cls[scene_id] = {}
            aggr_json = json.load(open(os.path.join(CONF.SCANNET_DIR, scene_id, "{}.aggregation.json".format(scene_id))))
            for entry in aggr_json["segGroups"]:
                object_id = int(entry["objectId"])
                object_name = entry["label"]
                sem_cls = self.raw2label[object_name] if object_name in self.raw2label else 17

                object_id_to_sem_cls[scene_id][object_id] = sem_cls # int to int

        return object_id_to_sem_cls
