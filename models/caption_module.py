import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.config import CONF
from utils.box_util import box3d_iou_batch_tensor

# constants
DC = ScannetDatasetConfig()

def select_target(data_dict):
    # predicted bbox
    pred_bbox = data_dict["bbox_corner"] # batch_size, num_proposals, 8, 3
    batch_size, num_proposals, _, _ = pred_bbox.shape

    # ground truth bbox
    gt_bbox = data_dict["ref_box_corner_label"] # batch_size, 8, 3

    target_ids = []
    target_ious = []
    for i in range(batch_size):
        # convert the bbox parameters to bbox corners
        pred_bbox_batch = pred_bbox[i] # num_proposals, 8, 3
        gt_bbox_batch = gt_bbox[i].unsqueeze(0).repeat(num_proposals, 1, 1) # num_proposals, 8, 3
        ious = box3d_iou_batch_tensor(pred_bbox_batch, gt_bbox_batch)
        target_id = ious.argmax().item() # 0 ~ num_proposals - 1
        target_ids.append(target_id)
        target_ious.append(ious[target_id])

    target_ids = torch.LongTensor(target_ids).cuda() # batch_size
    target_ious = torch.FloatTensor(target_ious).cuda() # batch_size

    return target_ids, target_ious

class SceneCaptionModule(nn.Module):
    def __init__(self, vocabulary, embeddings, emb_size=300, feat_size=128, hidden_size=512, num_proposals=256):
        super().__init__() 

        self.vocabulary = vocabulary
        self.embeddings = embeddings
        self.num_vocabs = len(vocabulary["word2idx"])

        self.emb_size = emb_size
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.num_proposals = num_proposals

        # transform the visual signals
        self.map_feat = nn.Sequential(
            nn.Linear(feat_size, emb_size),
            nn.ReLU()
        )

        # captioning core
        self.recurrent_cell = nn.GRUCell(
            input_size=emb_size,
            hidden_size=emb_size
        )
        self.classifier = nn.Linear(emb_size, self.num_vocabs)

    def step(self, step_input, hidden):
        hidden = self.recurrent_cell(step_input, hidden) # num_proposals, emb_size

        return hidden, hidden

    def forward(self, data_dict, use_tf=True, is_eval=False, max_len=CONF.TRAIN.MAX_DES_LEN):
        if not is_eval:
            data_dict = self.forward_sample_batch(data_dict, max_len)
        else:
            data_dict = self.forward_scene_batch(data_dict, use_tf, max_len)

        return data_dict

    def forward_sample_batch(self, data_dict, max_len=CONF.TRAIN.MAX_DES_LEN, min_iou=CONF.TRAIN.MIN_IOU_THRESHOLD):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        word_embs = data_dict["lang_feat"] # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"] # batch_size
        obj_feats = data_dict["bbox_feature"] # batch_size, num_proposals, feat_size
        
        num_words = des_lens.max()
        batch_size = des_lens.shape[0]

        # transform the features
        obj_feats = self.map_feat(obj_feats) # batch_size, num_proposals, emb_size

        # find the target object ids
        target_ids, target_ious = select_target(data_dict)

        # select object features
        target_feats = torch.gather(
            obj_feats, 1, target_ids.view(batch_size, 1, 1).repeat(1, 1, self.emb_size)).squeeze(1) # batch_size, emb_size

        # recurrent from 0 to max_len - 2
        outputs = []
        hidden = target_feats # batch_size, emb_size
        step_id = 0
        step_input = word_embs[:, step_id] # batch_size, emb_size
        while True:
            # feed
            step_output, hidden = self.step(step_input, hidden)
            step_output = self.classifier(step_output) # batch_size, num_vocabs
            
            # store
            step_output = step_output.unsqueeze(1) # batch_size, 1, num_vocabs 
            outputs.append(step_output)

            # next step
            step_id += 1
            if step_id == num_words - 1: break # exit for train mode
            step_input = word_embs[:, step_id] # batch_size, emb_size

        outputs = torch.cat(outputs, dim=1) # batch_size, num_words - 1/max_len, num_vocabs

        # NOTE when the IoU of best matching predicted boxes and the GT boxes 
        # are smaller than the threshold, the corresponding predicted captions
        # should be filtered out in case the model learns wrong things
        good_bbox_masks = target_ious > min_iou # batch_size
        # good_bbox_masks = target_ious != 0 # batch_size

        num_good_bboxes = good_bbox_masks.sum()
        mean_target_ious = target_ious[good_bbox_masks].mean() if num_good_bboxes > 0 else torch.zeros(1)[0].cuda()

        # store
        data_dict["lang_cap"] = outputs
        data_dict["pred_ious"] = mean_target_ious
        data_dict["good_bbox_masks"] = good_bbox_masks

        return data_dict


    def forward_scene_batch(self, data_dict, use_tf=False, max_len=CONF.TRAIN.MAX_DES_LEN):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        word_embs = data_dict["lang_feat"] # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"] # batch_size
        obj_feats = data_dict["bbox_feature"] # batch_size, num_proposals, feat_size
        
        num_words = des_lens.max()
        batch_size = des_lens.shape[0]

        # transform the features
        obj_feats = self.map_feat(obj_feats) # batch_size, num_proposals, emb_size

        # recurrent from 0 to max_len - 2
        outputs = []
        for prop_id in range(self.num_proposals):
            # select object features
            target_feats = obj_feats[:, prop_id] # batch_size, emb_size

            # start recurrence
            prop_outputs = []
            hidden = target_feats # batch_size, emb_size
            step_id = 0
            step_input = word_embs[:, step_id] # batch_size, emb_size
            while True:
                # feed
                step_output, hidden = self.step(step_input, hidden)
                step_output = self.classifier(step_output) # batch_size, num_vocabs
                
                # predicted word
                step_preds = []
                for batch_id in range(batch_size):
                    idx = step_output[batch_id].argmax() # 0 ~ num_vocabs
                    word = self.vocabulary["idx2word"][str(idx.item())]
                    emb = torch.FloatTensor(self.embeddings[word]).unsqueeze(0).cuda() # 1, emb_size
                    step_preds.append(emb)

                step_preds = torch.cat(step_preds, dim=0) # batch_size, emb_size

                # store
                step_output = step_output.unsqueeze(1) # batch_size, 1, num_vocabs 
                prop_outputs.append(step_output)

                # next step
                step_id += 1
                if not use_tf and step_id == max_len - 1: break # exit for eval mode
                if use_tf and step_id == num_words - 1: break # exit for train mode
                step_input = step_preds if not use_tf else word_embs[:, step_id] # batch_size, emb_size

            prop_outputs = torch.cat(prop_outputs, dim=1).unsqueeze(1) # batch_size, 1, num_words - 1/max_len, num_vocabs
            outputs.append(prop_outputs)

        outputs = torch.cat(outputs, dim=1) # batch_size, num_proposals, num_words - 1/max_len, num_vocabs

        # store
        data_dict["lang_cap"] = outputs

        return data_dict

class TopDownSceneCaptionModule(nn.Module):
    def __init__(self, vocabulary, embeddings, emb_size=300, feat_size=128, hidden_size=512, num_proposals=256, 
        num_locals=-1, query_mode="corner", use_relation=False, use_oracle=False):
        super().__init__() 

        self.vocabulary = vocabulary
        self.embeddings = embeddings
        self.num_vocabs = len(vocabulary["word2idx"])

        self.emb_size = emb_size
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.num_proposals = num_proposals
        
        self.num_locals = num_locals
        self.query_mode = query_mode

        self.use_relation = use_relation
        # if self.use_relation: self.map_rel = nn.Linear(feat_size * 2, feat_size)

        self.use_oracle = use_oracle

        # top-down recurrent module
        self.map_topdown = nn.Sequential(
            nn.Linear(hidden_size + feat_size + emb_size, emb_size),
            nn.ReLU()
        )
        self.recurrent_cell_1 = nn.GRUCell(
            input_size=emb_size,
            hidden_size=hidden_size
        )

        # top-down attention module
        self.map_feat = nn.Linear(feat_size, hidden_size, bias=False)
        self.map_hidd = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attend = nn.Linear(hidden_size, 1, bias=False)

        # language recurrent module
        self.map_lang = nn.Sequential(
            nn.Linear(feat_size + hidden_size, emb_size),
            nn.ReLU()
        )
        self.recurrent_cell_2 = nn.GRUCell(
            input_size=emb_size,
            hidden_size=hidden_size
        )
        self.classifier = nn.Linear(hidden_size, self.num_vocabs)

    def _step(self, step_input, target_feat, obj_feats, hidden_1, hidden_2, object_masks):
        '''
            recurrent step

            Args:
                step_input: current word embedding, (batch_size, emb_size)
                target_feat: object feature of the target object, (batch_size, feat_size)
                obj_feats: object features of all detected objects, (batch_size, num_proposals, feat_size)
                hidden_1: hidden state of top-down recurrent unit, (batch_size, hidden_size)
                hidden_2: hidden state of language recurrent unit, (batch_size, hidden_size)

            Returns:
                hidden_1: hidden state of top-down recurrent unit, (batch_size, hidden_size)
                hidden_2: hidden state of language recurrent unit, (batch_size, hidden_size)
                masks: attention masks on proposals, (batch_size, num_proposals, 1)
        '''

        # fuse inputs for top-down module
        step_input = torch.cat([step_input, hidden_2, target_feat], dim=-1)
        step_input = self.map_topdown(step_input)

        # top-down recurrence
        hidden_1 = self.recurrent_cell_1(step_input, hidden_1)

        # top-down attention
        combined = self.map_feat(obj_feats) # batch_size, num_proposals, hidden_size
        combined += self.map_hidd(hidden_1).unsqueeze(1) # batch_size, num_proposals, hidden_size
        combined = torch.tanh(combined)
        scores = self.attend(combined) # batch_size, num_proposals, 1
        scores.masked_fill_(object_masks == 0, float('-1e30'))

        masks = F.softmax(scores, dim=1) # batch_size, num_proposals, 1
        attended = obj_feats * masks
        attended = attended.sum(1) # batch_size, feat_size

        # fuse inputs for language module
        lang_input = torch.cat([attended, hidden_1], dim=-1)
        lang_input = self.map_lang(lang_input)

        # language recurrence
        hidden_2 = self.recurrent_cell_2(lang_input, hidden_2) # num_proposals, hidden_size

        return hidden_1, hidden_2, masks

    def _nn_distance(self, pc1, pc2):
        """
        Input:
            pc1: (B,N,C) torch tensor
            pc2: (B,M,C) torch tensor

        Output:
            dist1: (B,N) torch float32 tensor
            idx1: (B,N) torch int64 tensor
            dist2: (B,M) torch float32 tensor
            idx2: (B,M) torch int64 tensor
        """

        N = pc1.shape[1]
        M = pc2.shape[1]
        pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
        pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
        pc_diff = pc1_expand_tile - pc2_expand_tile
        pc_dist = torch.sqrt(torch.sum(pc_diff**2, dim=-1) + 1e-8) # (B,N,M)

        return pc_dist
    
    def _get_bbox_centers(self, corners):
        coord_min = torch.min(corners, dim=2)[0] # batch_size, num_proposals, 3
        coord_max = torch.max(corners, dim=2)[0] # batch_size, num_proposals, 3

        return (coord_min + coord_max) / 2

    def _query_locals(self, data_dict, target_ids, object_masks, include_self=True, overlay_threshold=CONF.TRAIN.OVERLAID_THRESHOLD):
        corners = data_dict["bbox_corner"] # batch_size, num_proposals, 8, 3
        centers = self._get_bbox_centers(corners) # batch_size, num_proposals, 3
        batch_size, _, _ = centers.shape

        # decode target box info
        target_centers = torch.gather(centers, 1, target_ids.view(-1, 1, 1).repeat(1, 1, 3)) # batch_size, 1, 3
        target_corners = torch.gather(corners, 1, target_ids.view(-1, 1, 1, 1).repeat(1, 1, 8, 3)) # batch_size, 1, 8, 3

        # get the distance
        if self.query_mode == "center":
            pc_dist = self._nn_distance(target_centers, centers).squeeze(1) # batch_size, num_proposals
        elif self.query_mode == "corner":
            pc_dist = self._nn_distance(target_corners.squeeze(1), centers) # batch_size, 8, num_proposals
            pc_dist, _ = torch.min(pc_dist, dim=1) # batch_size, num_proposals
        else:
            raise ValueError("invalid distance mode, choice: [\"center\", \"corner\"]")

        # mask out invalid objects
        pc_dist.masked_fill_(object_masks == 0, float('1e30')) # distance to invalid objects: infinity

        # exclude overlaid boxes
        tar2neigbor_iou = box3d_iou_batch_tensor(
            target_corners.repeat(1, self.num_proposals, 1, 1).view(-1, 8, 3), corners.view(-1, 8, 3)).view(batch_size, self.num_proposals) # batch_size, num_proposals
        overlaid_masks = tar2neigbor_iou >= overlay_threshold
        pc_dist.masked_fill_(overlaid_masks, float('1e30')) # distance to overlaid objects: infinity

        # include the target objects themselves
        self_dist = 0 if include_self else float('1e30')
        self_masks = torch.zeros(batch_size, self.num_proposals).cuda()
        self_masks.scatter_(1, target_ids.view(-1, 1), 1)
        pc_dist.masked_fill_(self_masks == 1, self_dist) # distance to themselves: 0 or infinity

        # get the top-k object ids
        _, topk_ids = torch.topk(pc_dist, self.num_locals, largest=False, dim=1) # batch_size, num_locals

        # construct masks for the local context
        local_masks = torch.zeros(batch_size, self.num_proposals).cuda()
        local_masks.scatter_(1, topk_ids, 1)

        return local_masks

    def forward(self, data_dict, use_tf=True, is_eval=False, max_len=CONF.TRAIN.MAX_DES_LEN):
        if not is_eval:
            data_dict = self._forward_sample_batch(data_dict, max_len)
        else:
            data_dict = self._forward_scene_batch(data_dict, use_tf, max_len)

        return data_dict

    def _create_adjacent_mat(self, data_dict, object_masks):
        batch_size, num_objects = object_masks.shape
        adjacent_mat = torch.zeros(batch_size, num_objects, num_objects).cuda()

        for obj_id in range(num_objects):
            target_ids = torch.LongTensor([obj_id for _ in range(batch_size)]).cuda()
            adjacent_entry = self._query_locals(data_dict, target_ids, object_masks, include_self=False) # batch_size, num_objects
            adjacent_mat[:, obj_id] = adjacent_entry

        return adjacent_mat

    def _get_valid_object_masks(self, data_dict, target_ids, object_masks):
        if self.num_locals == -1:
            valid_masks = object_masks
        else:
            adjacent_mat = data_dict["adjacent_mat"]
            batch_size, _, _ = adjacent_mat.shape
            valid_masks = torch.gather(
                adjacent_mat, 1, target_ids.view(batch_size, 1, 1).repeat(1, 1, self.num_proposals)).squeeze(1) # batch_size, num_proposals

        return valid_masks

    def _add_relation_feat(self, data_dict, obj_feats, target_ids):
        rel_feats = data_dict["edge_feature"] # batch_size, num_proposals, num_locals, feat_size
        batch_size = rel_feats.shape[0]

        rel_feats = torch.gather(rel_feats, 1, 
            target_ids.view(batch_size, 1, 1, 1).repeat(1, 1, self.num_locals, self.feat_size)).squeeze(1) # batch_size, num_locals, feat_size

        # new_obj_feats = torch.cat([obj_feats, rel_feats], dim=1) # batch_size, num_proposals + num_locals, feat_size

        # scatter the relation features to objects
        adjacent_mat = data_dict["adjacent_mat"] # batch_size, num_proposals, num_proposals
        rel_indices = torch.gather(adjacent_mat, 1, 
            target_ids.view(batch_size, 1, 1).repeat(1, 1, self.num_proposals)).squeeze(1) # batch_size, num_proposals
        rel_masks = rel_indices.unsqueeze(-1).repeat(1, 1, self.feat_size) == 1 # batch_size, num_proposals, feat_size
        scattered_rel_feats = torch.zeros(obj_feats.shape).cuda().masked_scatter(rel_masks, rel_feats) # batch_size, num_proposals, feat_size

        new_obj_feats = obj_feats + scattered_rel_feats
        # new_obj_feats = torch.cat([obj_feats, scattered_rel_feats], dim=-1)
        # new_obj_feats = self.map_rel(new_obj_feats)

        return new_obj_feats

    def _expand_object_mask(self, data_dict, object_masks, num_extra):
        batch_size, num_objects = object_masks.shape
        exp_masks = torch.zeros(batch_size, num_extra).cuda()

        num_edge_targets = data_dict["num_edge_target"]
        for batch_id in range(batch_size):
            exp_masks[batch_id, :num_edge_targets[batch_id]] = 1

        object_masks = torch.cat([object_masks, exp_masks], dim=1) # batch_size, num_objects + num_extra

        return object_masks

    def _forward_sample_batch(self, data_dict, max_len=CONF.TRAIN.MAX_DES_LEN, min_iou=CONF.TRAIN.MIN_IOU_THRESHOLD):
        """
            generate descriptions based on input tokens and object features
        """

        # unpack
        word_embs = data_dict["lang_feat"] # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"] # batch_size
        obj_feats = data_dict["bbox_feature"] # batch_size, num_proposals, feat_size
        object_masks = data_dict["bbox_mask"] # batch_size, num_proposals  
        
        num_words = des_lens.max()
        batch_size = des_lens.shape[0]

        # find the target object ids
        if self.use_oracle:
            target_ids = data_dict["bbox_idx"] # batch_size
            target_ious = torch.ones(batch_size).cuda()
        else:
            target_ids, target_ious = select_target(data_dict)

        # select object features
        target_feats = torch.gather(obj_feats, 1, target_ids.view(batch_size, 1, 1).repeat(1, 1, self.feat_size)).squeeze(1) # batch_size, emb_size

        # valid object proposal masks
        valid_masks = object_masks if self.num_locals == -1 else self._query_locals(data_dict, target_ids, object_masks)

        # object-to-object relation
        if self.use_relation:
            obj_feats = self._add_relation_feat(data_dict, obj_feats, target_ids)
            # valid_masks = self._expand_object_mask(data_dict, valid_masks, self.num_locals)

        # recurrent from 0 to max_len - 2
        outputs = []
        masks = []
        hidden_1 = torch.zeros(batch_size, self.hidden_size).cuda() # batch_size, hidden_size
        hidden_2 = torch.zeros(batch_size, self.hidden_size).cuda() # batch_size, hidden_size
        step_id = 0
        step_input = word_embs[:, step_id] # batch_size, emb_size
        while True:
            # feed
            hidden_1, hidden_2, step_mask = self._step(step_input, target_feats, obj_feats, hidden_1, hidden_2, valid_masks.unsqueeze(-1))
            step_output = self.classifier(hidden_2) # batch_size, num_vocabs
            
            # store
            step_output = step_output.unsqueeze(1) # batch_size, 1, num_vocabs 
            outputs.append(step_output)
            masks.append(step_mask) # batch_size, num_proposals, 1

            # next step
            step_id += 1
            if step_id == num_words - 1: break # exit for train mode
            step_input = word_embs[:, step_id] # batch_size, emb_size

        outputs = torch.cat(outputs, dim=1) # batch_size, num_words - 1/max_len, num_vocabs
        masks = torch.cat(masks, dim=-1) # batch_size, num_proposals, num_words - 1/max_len

        # NOTE when the IoU of best matching predicted boxes and the GT boxes 
        # are smaller than the threshold, the corresponding predicted captions
        # should be filtered out in case the model learns wrong things
        good_bbox_masks = target_ious > min_iou # batch_size

        num_good_bboxes = good_bbox_masks.sum()
        mean_target_ious = target_ious[good_bbox_masks].mean() if num_good_bboxes > 0 else torch.zeros(1)[0].cuda()

        # store
        data_dict["lang_cap"] = outputs
        data_dict["pred_ious"] = mean_target_ious
        data_dict["topdown_attn"] = masks
        data_dict["valid_masks"] = valid_masks
        data_dict["good_bbox_masks"] = good_bbox_masks

        return data_dict

    def _forward_scene_batch(self, data_dict, use_tf=False, max_len=CONF.TRAIN.MAX_DES_LEN):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        word_embs = data_dict["lang_feat"] # batch_size, emb_size
        obj_feats = data_dict["bbox_feature"] # batch_size, num_proposals, feat_size
        
        batch_size = word_embs.shape[0]

        # valid object proposal masks
        object_masks = data_dict["bbox_mask"]

        # # create adjacency matrices
        # if self.num_locals != -1 and "adjacent_mat" not in data_dict:
        #     adjacent_mat = self._create_adjacent_mat(data_dict, object_masks)
        #     data_dict["adjacent_mat"] = adjacent_mat
        
        # # include self to adjacency matrices
        # identity = torch.eye(self.num_proposals).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        # data_dict["adjacent_mat"] += identity # include self

        # recurrent from 0 to max_len - 2
        outputs = []
        masks = []
        valid_masks = []
        for prop_id in range(self.num_proposals):
            # select object features
            target_feats = obj_feats[:, prop_id] # batch_size, emb_size
            target_ids = torch.zeros(batch_size).fill_(prop_id).long().cuda()

            prop_obj_feats = obj_feats.clone()
            # valid_prop_masks = self._get_valid_object_masks(data_dict, target_ids, object_masks)
            valid_prop_masks = object_masks if self.num_locals == -1 else self._query_locals(data_dict, target_ids, object_masks)

            # object-to-object relation
            if self.use_relation:
                prop_obj_feats = self._add_relation_feat(data_dict, prop_obj_feats, target_ids)
                # valid_prop_masks = self._expand_object_mask(data_dict, valid_prop_masks, self.num_locals)

            valid_masks.append(valid_prop_masks.unsqueeze(1))

            # start recurrence
            prop_outputs = []
            prop_masks = []
            hidden_1 = torch.zeros(batch_size, self.hidden_size).cuda() # batch_size, hidden_size
            hidden_2 = torch.zeros(batch_size, self.hidden_size).cuda() # batch_size, hidden_size
            step_id = 0
            # step_input = word_embs[:, step_id] # batch_size, emb_size
            step_input = word_embs[:, 0] # batch_size, emb_size
            while True:
                # feed
                hidden_1, hidden_2, step_mask = self._step(step_input, target_feats, prop_obj_feats, hidden_1, hidden_2, valid_prop_masks.unsqueeze(-1))
                step_output = self.classifier(hidden_2) # batch_size, num_vocabs
                
                # predicted word
                step_preds = []
                for batch_id in range(batch_size):
                    idx = step_output[batch_id].argmax() # 0 ~ num_vocabs
                    word = self.vocabulary["idx2word"][str(idx.item())]
                    emb = torch.FloatTensor(self.embeddings[word]).unsqueeze(0).cuda() # 1, emb_size
                    step_preds.append(emb)

                step_preds = torch.cat(step_preds, dim=0) # batch_size, emb_size

                # store
                step_output = step_output.unsqueeze(1) # batch_size, 1, num_vocabs 
                prop_outputs.append(step_output)
                prop_masks.append(step_mask)

                # next step
                step_id += 1
                if step_id == max_len - 1: break # exit for eval mode
                step_input = step_preds # batch_size, emb_size

            prop_outputs = torch.cat(prop_outputs, dim=1).unsqueeze(1) # batch_size, 1, num_words - 1/max_len, num_vocabs
            prop_masks = torch.cat(prop_masks, dim=-1).unsqueeze(1) # batch_size, 1, num_proposals, num_words - 1/max_len
            outputs.append(prop_outputs)
            masks.append(prop_masks)

        outputs = torch.cat(outputs, dim=1) # batch_size, num_proposals, num_words - 1/max_len, num_vocabs
        masks = torch.cat(masks, dim=1) # batch_size, num_proposals, num_proposals, num_words - 1/max_len
        valid_masks = torch.cat(valid_masks, dim=1) # batch_size, num_proposals, num_proposals

        # store
        data_dict["lang_cap"] = outputs
        data_dict["topdown_attn"] = masks
        data_dict["valid_masks"] = valid_masks

        return data_dict
