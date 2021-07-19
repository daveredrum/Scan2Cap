import os
import random
import time
import json
import math
import torch
import h5py
import imagesize
import numpy as np
from tqdm import tqdm
import pickle5 as pickle
from itertools import chain
from collections import Counter
from torch.utils.data import Dataset
import torch.utils.data as data_tools
from itertools import permutations
import pdb

class ScanReferDataset(Dataset):

    def __init__(self,
                 split,
                 sample_list,
                 scene_list,
                 run_config
                 ):
        self.split = split
        self.sample_list = sample_list
        self.scene_list = scene_list
        self.run_config = run_config
        self.vocabulary = None
        self.weights = None
        self.glove = None
        self.lang = None
        self.lang_ids = None
        self.pre_training_verification()
        self.id2name = self.get_id2name_file()
        self.load_data()

    def pre_training_verification(self):
        self.prepare_db()
        self.verify_keys()
        self.update_samples()
        self.db.close()

    def prepare_db(self):
        assert os.path.exists(self.run_config.PATH.DB_PATH)
        self.db = h5py.File(self.run_config.PATH.DB_PATH, 'r')
    
    def purposefully_ignored_keys(self):
        ignored_samples = json.load(open(self.run_config.PATH.IGNORED_SAMPLES))
        return ignored_samples

    def verify_keys(self):
        # Part 1
        target_sample_keys = ['{}-{}_{}'.format(item['scene_id'], item['object_id'], item['ann_id']) for item in self.sample_list]
        db_keys = list(self.db['box'].keys())
        ignored_keys = [k for k in target_sample_keys if k not in db_keys]
        # Part 2
        ignored_keys += self.purposefully_ignored_keys()
        print("Number of ignored keys: {}".format(len(ignored_keys)))
        # assert len(ignored_keys) < 3000   # problematic keys
        self.ignored_keys = ignored_keys

    def update_samples(self):
        updated_sample_list = []
        for sample in self.sample_list:
            kf = '{}-{}_{}'.format(sample['scene_id'], sample['object_id'], sample['ann_id'])
            if kf not in self.ignored_keys:
                updated_sample_list.append(sample)
        
        self.verified_list = updated_sample_list

        print("Number of samples before ignoring: ", len(self.sample_list))
        print("Number of samples after ignoring: ", len(self.verified_list))
        # print("Ignored keys: ", self.ignored_keys)

    def __len__(self):
        return len(self.verified_list)

    def __getitem__(self, idx):
        start = time.time()
        item = self.verified_list[idx]
        scene_id = item['scene_id']
        target_id = item['object_id']
        ann_id = item['ann_id']
        sample_id = '{}-{}_{}'.format(scene_id, target_id, ann_id)
        lang_feat = self.lang[sample_id]
        lang_ids = np.array(self.lang_ids[sample_id])
        lang_len = len(item["token"]) + 2
        lang_len = lang_len if lang_len <= self.run_config.MAX_DESC_LEN + 2 else self.run_config.MAX_DESC_LEN + 2

        with h5py.File(self.run_config.PATH.DB_PATH, 'r') as db:
            boxes = np.array(db['box'][sample_id])
            box_feats = np.array(db['boxfeat'][sample_id])
            object_ids = np.array(db['objectids'][sample_id])
            global_feat = np.array(db['globalfeat'][sample_id])
            target_idx = np.where(object_ids == int(target_id))[0]
            if target_idx.shape[0] == 1:
                target_feat = np.concatenate((box_feats[target_idx], boxes[target_idx]), axis=1)
                pool_feats = np.concatenate((box_feats, boxes), axis=1)
                pool_ids = object_ids

            else:
                random_idx = 0  # sorted samples in mrcnn mode
                target_feat = np.concatenate((box_feats[random_idx][None, :], boxes[random_idx][None, :]), axis=1)
                pool_feats = np.concatenate((box_feats, boxes), axis=1)
                pool_ids = object_ids

            ret = {
                'failed': False,
                'lang_feat': lang_feat,
                'lang_len': lang_len,
                'lang_ids': lang_ids,
                't_feat': target_feat,
                't_id': np.array(target_id, dtype=np.int16),
                'c_feats': pool_feats,
                'c_ids': pool_ids,
                'g_feat': global_feat,
                'sample_id': sample_id,
                'load_time': time.time() - start
            }

        return ret

    def get_raw2label(self):
        # mapping
        scannet_labels = self.run_config.LABEL2CLASS
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(self.run_config.PATH.SCANNET_V2_TSV)]
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

    def get_id2name_file(self):
        print("getting id2name...")
        id2name = {}
        item_ids = []
        for scene_id in tqdm(self.scene_list):
            id2name[scene_id] = {}
            aggr_file = json.load(open(self.run_config.PATH.AGGR_JSON.format(scene_id, scene_id)))
            for item in aggr_file["segGroups"]:
                item_ids.append(int(item["id"]))
                id2name[scene_id][int(item["id"])] = item["label"]

        return id2name

    def get_label_info(self):
        label2class = self.run_config.LABEL2CLASS

        # mapping
        scannet_labels = label2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}
        lines = [line.rstrip() for line in open(self.run_config.PATH.SCANNET_V2_TSV)]
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

        return raw2label, label2class

    def transform_des(self):
        lang = {}
        label = {}
        max_len = self.run_config.MAX_DESC_LEN
        for data in self.sample_list:
            scene_id = data['scene_id']
            object_id = data['object_id']
            ann_id = data['ann_id']
            sample_id = '{}-{}_{}'.format(scene_id, object_id, ann_id)

            if sample_id not in lang:
                lang[sample_id] = {}
                label[sample_id] = {}

            # trim long descriptions
            tokens = data["token"][:max_len]

            # tokenize the description
            tokens = ["sos"] + tokens + ["eos"]
            embeddings = np.zeros((max_len + 2, 300))
            labels = np.zeros((max_len + 2))  # start and end

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
            lang[sample_id] = embeddings
            label[sample_id] = labels

        return lang, label

    def build_vocab(self):
        if os.path.exists(self.run_config.PATH.SCANREFER_VOCAB):
            self.vocabulary = json.load(open(self.run_config.PATH.SCANREFER_VOCAB))
            print("Loaded the existing vocabulary.")
        else:
            print("Resetting vocabulary.")
            if self.split == "train":
                all_words = chain(*[data["token"][:self.run_config.MAX_DESC_LEN] for data in self.sample_list])
                word_counter = Counter(all_words)
                word_counter = sorted([(k, v) for k, v in word_counter.items() if k in self.glove],
                                      key=lambda x: x[1],
                                      reverse=True)
                word_list = [k for k, _ in word_counter]

                # build vocabulary
                word2idx, idx2word = {}, {}
                spw = ["pad_", "unk", "sos",
                       "eos"]  # NOTE distinguish padding token "pad_" and the actual word "pad"
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
                json.dump(vocab, open(self.run_config.PATH.SCANREFER_VOCAB, "w"), indent=4)

                self.vocabulary = vocab

        print("Number of keys in the vocab: {}".format(len(self.vocabulary['idx2word'].keys())))

    def build_frequency(self):
        if os.path.exists(self.run_config.PATH.SCANREFER_VOCAB_WEIGHTS):
            with open(self.run_config.PATH.SCANREFER_VOCAB_WEIGHTS) as f:
                weights = json.load(f)
                self.weights = np.array([v for _, v in weights.items()])
        else:

            all_tokens = []

            for sample_id in self.lang_ids.keys():
                all_tokens += self.lang_ids[sample_id].astype(int).tolist()

            word_count = Counter(all_tokens)
            word_count = sorted([(k, v) for k, v in word_count.items()], key=lambda x: x[0])
            weights = np.ones((len(word_count)))
            self.weights = weights

            with open(self.run_config.PATH.SCANREFER_VOCAB_WEIGHTS, "w") as f:
                weights = {k: v for k, v in enumerate(weights)}
                json.dump(weights, f, indent=4)

    def load_data(self):
        print("loading data...")
        # load language features
        self.glove = pickle.load(open(self.run_config.PATH.GLOVE_PICKLE, "rb"))
        self.build_vocab()
        self.num_vocabs = len(self.vocabulary["word2idx"].keys())
        self.lang, self.lang_ids = self.transform_des()
        self.build_frequency()

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.sample_list])))

        # prepare class mapping
        lines = [line.rstrip() for line in open(self.run_config.PATH.SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self.get_raw2label()

    def collate_fn(self, data):
        data = list(filter(lambda x: x['failed'] == False, data))
        data_dicts = sorted(data, key=lambda d: d['c_ids'].shape[0], reverse=True)
        max_proposals_in_batch = data_dicts[0]['c_ids'].shape[0]
        batch_size = len(data_dicts)
        lang_feat = torch.zeros((batch_size, len(data_dicts[0]['lang_feat']), len(data_dicts[0]['lang_feat'][0])),
                                dtype=torch.float32)
        lang_len = torch.zeros((batch_size, 1), dtype=torch.int16)
        lang_ids = torch.zeros((batch_size, len(data_dicts[0]['lang_ids'])), dtype=torch.long)
        sample_ids = []
        vis_feats = torch.zeros((batch_size, self.run_config.GLOBAL_FEATURE_SIZE), dtype=torch.float)
        target_feat = torch.zeros((batch_size, self.run_config.TARGET_FEATURE_SIZE), dtype=torch.float)
        target_object_id = torch.zeros((batch_size, 1), dtype=torch.int16)
        padded_proposal_feat = torch.zeros((batch_size, max_proposals_in_batch, self.run_config.PROPOSAL_FEATURE_SIZE))
        padded_proposal_object_ids = torch.zeros((batch_size, max_proposals_in_batch, 1), dtype=torch.int16)
        padded_proposal_object_ids[:, :, :] = -1
        times = torch.zeros((batch_size, 1))

        for ix, d in enumerate(data_dicts):
            num_proposals = len(d['c_ids'])
            padded_proposal_feat[ix, :num_proposals, :] = torch.from_numpy(d['c_feats']).unsqueeze(0)
            padded_proposal_object_ids[ix, :num_proposals, :] = torch.from_numpy(d['c_ids'])
            vis_feats[ix, :] = torch.from_numpy(d['g_feat']).squeeze().unsqueeze(0)
            target_feat[ix, :] = torch.from_numpy(d['t_feat']).squeeze().unsqueeze(0)
            target_object_id[ix, :] = torch.from_numpy((d['t_id']))
            lang_feat[ix, :] = torch.tensor(d['lang_feat'])
            lang_len[ix, :] = torch.tensor(d['lang_len'])
            lang_ids[ix, :] = torch.tensor(d['lang_ids'])
            sample_ids.append(d['sample_id'])
            times[ix, :] = d['load_time']

        return {
            'lang_feat': lang_feat,
            'lang_len': lang_len,
            'lang_ids': lang_ids,
            't_feat': target_feat,
            't_id': target_object_id,
            'c_feats': padded_proposal_feat,
            'c_ids': padded_proposal_object_ids,
            'g_feat': vis_feats,
            'sample_id': sample_ids,
            'load_time': times
        }

    def get_candidate_extras(self, candidates):
        """
            generates a list, containing the predictions on validation set with the following format.
            [
                {
                    "scene_id": "...",
                    "object_id": "...",
                    "ann_id": "...",
                    "camera_pose": "...",
                    "description": "1 caption here"
                    "bbox_corner": ["x_min", "y_min", "x_max", "y_max"],
                    "object_mask": "RLE FORMAT DETECTED OBJECT MASK",
                    "depth_file_name": "..."
                }
            ]
        :param candidates: dictionary mapping from keys to captions.
        :return: info_list, described above.
        """

        info_list = []
        for sample_id, caption in candidates.items():
            with open(self.run_config.PATH.BOX, 'rb') as f:
                box = pickle.load(f)
                box = box[sample_id]  # Returns a list of dict
            camera_pose = list(filter(
                lambda x: x['sample_id'] == sample_id, self.sample_list))
            assert len(camera_pose) == 1
            transformation = camera_pose[0]['transformation']
            bbox = box['bbox']
            mask = box['mask']
            depth_file_name = '{}.depth.png'.format(sample_id)
            info_dict = {
                'sample_id': sample_id,
                'transformation': transformation,
                'description': caption,
                'detected_bbox': bbox,
                'detected_mask_rle': mask,
                'depth_file_name': depth_file_name
            }
            info_list.append(info_dict)
        return info_list

