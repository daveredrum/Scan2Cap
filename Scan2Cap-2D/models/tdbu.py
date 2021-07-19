import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TDBUCaptionBase(nn.Module):
    def __init__(self,
                 device,
                 max_desc_len,
                 vocabulary,
                 embeddings,
                 emb_size=300,
                 feat_size=128,
                 hidden_size=512,
                 ):
        super().__init__()

        self.device = device
        self.max_desc_len = max_desc_len
        self.vocabulary = vocabulary
        self.embeddings = embeddings
        self.num_vocabs = len(vocabulary["word2idx"])

        self.emb_size = emb_size
        self.feat_size = feat_size
        self.hidden_size = hidden_size

        # top-down recurrent module
        bias_state = False
        self.map_topdown_1 = nn.Linear(hidden_size, 128, bias=bias_state)
        self.map_topdown_2 = nn.Linear(feat_size, 128, bias=bias_state)
        self.map_topdown_3 = nn.Linear(emb_size, 128, bias=bias_state)
        self.map_topdown = nn.Linear(128, 128, bias=bias_state)
        self.recurrent_cell_1 = nn.GRUCell(
            input_size=128,
            hidden_size=hidden_size
        )

        # top-down attention module
        self.map_feat = nn.Linear(feat_size, hidden_size, bias=bias_state)
        self.map_hidd = nn.Linear(hidden_size, hidden_size, bias=bias_state)
        self.attend = nn.Linear(hidden_size, 1, bias=bias_state)

        # language recurrent module
        self.map_lang_1 = nn.Linear(feat_size, 128, bias=bias_state)
        self.map_lang_2 = nn.Linear(hidden_size, 128, bias=bias_state)
        self.map_lang = nn.Linear(128, 128, bias=bias_state)
        self.recurrent_cell_2 = nn.GRUCell(
            input_size=128,
            hidden_size=hidden_size
        )
        self.classifier = nn.Linear(hidden_size, self.num_vocabs)

    def step(self, step_input, target_feat, obj_feats, hidden_1, hidden_2):
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
        step_input = self.map_topdown_3(step_input)  # batch * 128
        step_input += self.map_topdown_1(hidden_2)  # batch * 128
        step_input += self.map_topdown_2(target_feat)  # batch * 128
        step_input = torch.tanh(step_input)
        step_input = self.map_topdown(step_input)
        step_input = F.relu(step_input)

        # top-down recurrence
        hidden_1 = self.recurrent_cell_1(step_input, hidden_1)

        # top-down attention
        combined = self.map_feat(obj_feats)  # batch_size, num_proposals, hidden_size
        combined += self.map_hidd(hidden_1).unsqueeze(1)  # batch_size, num_proposals, hidden_size
        combined = torch.tanh(combined)
        scores = self.attend(combined)  # batch_size, num_proposals, 1
        masks = F.softmax(scores, dim=1)  # batch_size, num_proposals, 1
        attended = obj_feats * masks
        attended = attended.sum(1)  # batch_size, feat_size

        # fuse inputs for language module
        lang_input = self.map_lang_1(attended)
        lang_input += self.map_lang_2(hidden_1)
        lang_input = torch.tanh(lang_input)
        lang_input = self.map_lang(lang_input)
        lang_input = F.relu(lang_input)

        # language recurrence
        hidden_2 = self.recurrent_cell_2(lang_input, hidden_2)  # num_proposals, hidden_size

        return hidden_1, hidden_2, masks


class ShowAttendAndTell(TDBUCaptionBase):

    def __init__(self,
                 device,
                 max_desc_len,
                 vocabulary,
                 embeddings,
                 emb_size,
                 feat_size,
                 context_size,
                 feat_input,
                 hidden_size
                 ):

        self.max_desc_len = max_desc_len
        self.feat_size = feat_size
        self.context_size = context_size
        self.feat_input = feat_input
        super().__init__(
            device=device,
            max_desc_len=max_desc_len,
            vocabulary=vocabulary,
            embeddings=embeddings,
            emb_size=emb_size,
            feat_size=context_size,
            hidden_size=hidden_size
        )

        if self.feat_input['add_global']:
            self.reduce_dim = nn.Sequential(
                nn.Linear(in_features=feat_size, out_features=self.context_size),
                nn.ReLU()
            )

        assert self.feat_input['add_target']

    # def send_to_device(self, dd):
    #     dd_new = {}
    #     for k, v in dd.items():
    #         if isinstance(v, torch.Tensor):
    #             dd_new[k] = v.to(self.device)
            
    #     return dd_new
        
    def forward(self, data_dict, is_eval=False):

        if self.feat_input['add_global']:
            t_feat = data_dict['t_feat']  # batch_size, object_feat_size
            g_feat = data_dict['g_feat']  # batch_size, 2048
            t_feat = torch.cat((g_feat, t_feat), dim=1)  # batch_size
            t_feat = self.reduce_dim(t_feat)
            data_dict['inp_feat'] = t_feat
        else:
            data_dict['inp_feat'] = data_dict['t_feat']
            
        if not is_eval:
            # During training
            data_dict = self.forward_train_batch(data_dict)
        else:
            # During evaluation
            data_dict = self.forward_inference_batch(data_dict)

        return data_dict

    def forward_train_batch(self, data_dict):
        """
            generate descriptions based on input tokens and object features
        """
        word_embs = data_dict["lang_feat"]  # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"]  # batch_size
        c_feat = data_dict['c_feats']  # batch_size, num_objects_in_image, object_feat_size
        t_feat = data_dict['inp_feat']  # batch_size, object_feat_size

        # batch_size, object_feat_size
        t_feat = t_feat.squeeze()
        num_words = des_lens[0]
        batch_size = des_lens.shape[0]
        # recurrent from 0 to max_len - 2
        outputs = []
        masks = []
        hidden_1 = torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device)  # batch_size, hidden_size
        hidden_2 = torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device)  # batch_size, hidden_size
        step_id = 0
        step_input = word_embs[:, step_id]  # batch_size, emb_size
        while True:
            # feed
            hidden_1, hidden_2, step_mask = self.step(step_input, t_feat, c_feat, hidden_1, hidden_2)
            step_output = self.classifier(hidden_2)  # batch_size, num_vocabs

            # # predicted word
            step_preds = []
            for batch_id in range(batch_size):
                idx = step_output[batch_id].argmax()  # 0 ~ num_vocabs
                word = self.vocabulary["idx2word"][str(idx.item())]
                emb = torch.tensor(self.embeddings[word], dtype=torch.float).unsqueeze(0).to(self.device)
                step_preds.append(emb)

            # store
            step_output = step_output.unsqueeze(1)  # batch_size, 1, num_vocabs
            outputs.append(step_output)
            masks.append(step_mask)  # batch_size, 1

            # next step
            step_id += 1
            if step_id == num_words - 1:
                break  # exit for train mode
            step_input = word_embs[:, step_id]  # batch_size, emb_size

        outputs = torch.cat(outputs, dim=1)  # batch_size, num_words - 1/max_len, num_vocabs
        masks = torch.cat(masks, dim=-1)  # batch_size, num_words - 1/max_len

        data_dict["lang_cap"] = outputs
        data_dict["topdown_attn"] = masks

        return data_dict

    def forward_inference_batch(self, data_dict):
        """
        generate descriptions based on input tokens and object features
        """
        word_embs = data_dict["lang_feat"]  # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"]  # batch_size
        c_feat = data_dict["c_feats"]  # batch_size, num_proposals, object_feat_size
        batch_size = des_lens.shape[0]
        t_feat = data_dict["t_feat"]

        # recurrent from 0 to max_len - 2
        outputs = []
        masks = []

        t_feat = t_feat.squeeze()
        # start recurrence
        hidden_1 = torch.zeros(batch_size, self.hidden_size, requires_grad=False).to(self.device)  # batch_size, hidden_size
        hidden_2 = torch.zeros(batch_size, self.hidden_size, requires_grad=False).to(self.device)  # batch_size, hidden_size
        step_id = 0
        step_input = word_embs[:, step_id]  # batch_size, emb_size
        while True:
            # feed
            hidden_1, hidden_2, step_mask = self.step(step_input, t_feat, c_feat, hidden_1, hidden_2)
            step_output = self.classifier(hidden_2)  # batch_size, num_vocabs

            # predicted word
            step_preds = []
            for batch_id in range(batch_size):
                idx = step_output[batch_id].argmax()  # 0 ~ num_vocabs
                word = self.vocabulary["idx2word"][str(idx.item())]
                emb = torch.tensor(self.embeddings[word], dtype=torch.float).unsqueeze(0).to(self.device)
                step_preds.append(emb)

            step_preds = torch.cat(step_preds, dim=0)  # batch_size, emb_size

            # store
            step_output = step_output.unsqueeze(1)  # batch_size, 1, num_vocabs
            outputs.append(step_output)
            masks.append(step_mask)

            # next step
            step_id += 1
            if step_id == self.max_desc_len - 1:
                break  # exit for eval mode
            step_input = step_preds # batch_size, emb_size

        outputs = torch.cat(outputs, dim=1)  # batch_size, num_words - 1/max_len, num_vocabs
        masks = torch.cat(masks, dim=-1)  # batch_size, num_words - 1/max_len

        # store
        data_dict["lang_cap"] = outputs
        data_dict["topdown_attn"] = masks

        return data_dict
