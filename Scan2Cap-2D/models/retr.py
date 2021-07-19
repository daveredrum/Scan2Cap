import numpy as np
import torch.nn as nn
from numpy.linalg import norm


class Retrieval2D(nn.Module):
    def __init__(self, db_path, feat_size, vis_feat_dict, lang_ids):
        super().__init__()
        self.db_path = db_path
        self.feat_size = feat_size
        self.vis_feat_dict = vis_feat_dict
        self.n_repeats = len(list(self.vis_feat_dict.keys()))
        _ = self.create_train_memory_map()
        self.train_memory_map = self.read_train_memory_map()
        self.lang_ids = lang_ids

    def read_train_memory_map(self):
        return np.memmap(self.db_path, dtype="float32", mode="r", shape=(self.n_repeats, self.feat_size))

    def create_train_memory_map(self):
        train_map = np.memmap(self.db_path, dtype="float32", mode="w+", shape=(self.n_repeats, self.feat_size))
        for i, tns in enumerate(self.vis_feat_dict.values()):
            train_map[i, :] = tns

        return train_map

    def get_best_rank_id(self, vis_feat):
        val_memory_map = vis_feat.repeat(self.n_repeats, 1)
        train_memory_map = self.train_memory_map
        cosine_ranked = np.einsum('xy,xy->x', val_memory_map, train_memory_map) / (
                    norm(val_memory_map, axis=1) * norm(train_memory_map, axis=1))
        sample_id = list(self.vis_feat_dict.keys())[np.argmax(cosine_ranked)]
        return sample_id

    def forward(self, data_dict):
        vis_feats = data_dict['t_feat'][:, :-4]     # Ignore the concatenated bbox
        batch_size = vis_feats.shape[0]

        batch_captions = []
        for i in range(batch_size):
            # perform a global cosine ranking
            vis_feat = vis_feats[i, :]
            sample_id = self.get_best_rank_id(vis_feat)
            caption = self.lang_ids[sample_id]
            batch_captions.append(caption)

        return batch_captions  # batch_size, num_words
