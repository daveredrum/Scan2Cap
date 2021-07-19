import os
import torch
import json
import pickle
import math
import h5py
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


class FrameData(Dataset):
    def __init__(self, ignored_samples, key_format, resize, frame_path, db_path, box, input_list, transforms):
        self.ignored_samples = ignored_samples
        self.new_width, self.new_height = resize
        self.key_format = key_format
        self.frames_path = frame_path
        self.db_path = db_path
        self.box = box
        self.input_list = input_list
        self.transforms = transforms
        print("BOX: ", self.box)
        self.pre_training_verification()

    def pre_training_verification(self):
        self.prepare_db()
        self.verify_keys()
        self.update_samples()
        self.db.close()

    def prepare_db(self):
        assert os.path.exists(self.db_path)
        self.db = h5py.File(self.db_path, 'r')

    def purposefully_ignored_keys(self):
        ignored_samples = json.load(open(self.ignored_samples))
        return ignored_samples

    def verify_keys(self):
        # Part 1
        target_sample_keys = ['{}-{}_{}'.format(item['scene_id'], item['object_id'], item['ann_id']) for item in self.input_list]
        db_keys = list(self.db['box'].keys())
        ignored_keys = [k for k in target_sample_keys if k not in db_keys]
        # Part 2
        ignored_keys += self.purposefully_ignored_keys()
        print("Number of ignored keys: {}".format(len(ignored_keys)))
        # assert len(ignored_keys) < 3000   # problematic keys
        self.ignored_keys = ignored_keys

    def update_samples(self):
        updated_sample_list = []
        for sample in self.input_list:
            kf = '{}-{}_{}'.format(sample['scene_id'], sample['object_id'], sample['ann_id'])
            if kf not in self.ignored_keys:
                updated_sample_list.append(sample)
        
        self.verified_list = updated_sample_list

        print("Number of samples before ignoring: ", len(self.input_list))
        print("Number of samples after ignoring: ", len(self.verified_list))
        print("Ignored keys: ", self.ignored_keys)

    def __len__(self):
        return len(self.verified_list)

    def __getitem__(self, idx):
        input_item = self.verified_list[idx]
        scene_id = input_item['scene_id']
        object_id = input_item['object_id']
        ann_id = input_item['ann_id']
        sample_id = self.key_format.format(scene_id, object_id, ann_id)

        frame_tensor = self.load_image(
            image_path=os.path.join(self.frames_path, scene_id, '{}.png'.format(sample_id))
        )

        # load bbox info
        sample_id = '{}-{}_{}'.format(scene_id, object_id, ann_id)
        with h5py.File(self.db_path, 'r') as db:
            if self.box:
                boxes = np.array(db['box'][sample_id])
                bbox_ids = np.array(db['objectids'][sample_id])
                ret = {
                    'frame_tensor': frame_tensor,
                    'bbox_info': boxes,
                    'bbox_id': bbox_ids,
                    'sample_id': sample_id
                }

            else:
                ret = {
                    'frame_tensor': frame_tensor,
                    'bbox_info': None,
                    'bbox_id': None,
                    'sample_id': sample_id
                }

        return ret

    def load_image(self, image_path):
        old_image = Image.open(image_path)
        resized = old_image.resize((self.new_width, self.new_height))
        rgbed = resized.convert('RGB')
        frame_tensor = torch.from_numpy(np.asarray(rgbed).astype(np.float32)).permute(2, 0, 1)

        if self.transforms:
            frame_tensor = self.transforms(frame_tensor / 255.0)

        return frame_tensor

    def collate_fn(self, data):
        # data = list(filter(lambda x: x['use'], data))
        tensor_list = [d['frame_tensor'] for d in data]
        bbox_list = [d['bbox_info'] for d in data]
        bbox_ids = [d['bbox_id'] for d in data]
        sample_id_list = [d['sample_id'] for d in data]

        return tensor_list, bbox_list, bbox_ids, sample_id_list