import torch
import torch.nn as nn


class CaptionBase(nn.Module):
    def __init__(self,
                 device,
                 max_desc_len,
                 vocabulary,
                 embeddings,
                 emb_size,
                 feat_size,
                 hidden_size
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
        self.map_feat = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            nn.ReLU()
        )
        self.recurrent_cell = nn.GRUCell(
            input_size=emb_size,
            hidden_size=hidden_size
        )
        self.classifier = nn.Linear(hidden_size, self.num_vocabs)

    def step(self, step_input, hidden):
        hidden = self.recurrent_cell(step_input, hidden)

        return hidden, hidden


class ShowAndTell(CaptionBase):
    def __init__(self,
                 device,
                 max_desc_len,
                 vocabulary,
                 embeddings,
                 emb_size,
                 feat_size,
                 feat_input,
                 hidden_size
                 ):

        super().__init__(
            device=device,
            max_desc_len=max_desc_len,
            vocabulary=vocabulary,
            embeddings=embeddings,
            emb_size=emb_size,
            feat_size=feat_size,
            hidden_size=hidden_size
        )
        self.feat_input = feat_input
        assert self.feat_input['add_global']

    def forward(self, data_dict, is_eval=False):

        g_feat = data_dict["g_feat"]
        if self.feat_input['add_target']:
            t_feat = data_dict["t_feat"]
            t_feat = torch.cat((g_feat, t_feat), dim=1)
            data_dict['inp_feat'] = t_feat

        else:
            data_dict['inp_feat'] = g_feat

        if not is_eval:
            data_dict = self.forward_train_batch(data_dict)
        else:
            data_dict = self.forward_inference_batch(data_dict)

        return data_dict

    def forward_train_batch(self, data_dict):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        word_embs = data_dict["lang_feat"]  # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"]  # batch_size
        t_feat = data_dict["inp_feat"]
        num_words = des_lens[0]
        batch_size = des_lens.shape[0]
        t_feat = self.map_feat(t_feat.squeeze())
        
        # recurrent from 0 to max_len - 2
        outputs = []
        hidden = t_feat  # batch_size, emb_size
        step_id = 0
        step_input = word_embs[:, step_id, :]  # batch_size, emb_size
        while True:
            # feed
            step_output, hidden = self.step(step_input, hidden)
            step_output = self.classifier(step_output)  # batch_size, num_vocabs

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

            # next step
            step_id += 1
            if step_id == num_words - 1: break  # exit for train mode

            step_input = word_embs[:, step_id]  # batch_size, emb_size

        outputs = torch.cat(outputs, dim=1)  # batch_size, num_words - 1/max_len, num_vocabs

        # store
        data_dict["lang_cap"] = outputs

        return data_dict

    def forward_inference_batch(self, data_dict):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        word_embs = data_dict["lang_feat"]  # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"]  # batch_size
        t_feat = data_dict["inp_feat"]
        num_words = des_lens[0]
        batch_size = des_lens.shape[0]

        # transform the features
        t_feat = self.map_feat(t_feat.squeeze())  # batch_size, num_proposals, emb_size

        # recurrent from 0 to max_len - 2
        outputs = []

        # start recurrence
        hidden = t_feat  # batch_size, emb_size
        step_id = 0
        step_input = word_embs[:, step_id]  # batch_size, emb_size

        while True:
            # feed
            step_output, hidden = self.step(step_input, hidden)
            step_output = self.classifier(step_output)  # batch_size, num_vocabs

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

            # next step
            step_id += 1
            if step_id == self.max_desc_len - 1: break  # exit for eval mode
            step_input = step_preds

            outputs.append(step_output)

        outputs = torch.cat(outputs, dim=1)  # batch_size, num_words - 1/max_len, num_vocabs

        # store
        data_dict["lang_cap"] = outputs

        return data_dict
