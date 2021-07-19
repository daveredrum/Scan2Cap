import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder

class CapNet(nn.Module):
    def __init__(self, mode, vocabulary, embeddings, use_topdown=False, num_locals=-1, query_mode="corner",
        graph_mode="graph_conv", num_graph_steps=0, use_relation=False, graph_aggr="add",
        use_orientation=False, num_bins=6, use_distance=False,
        emb_size=300, hidden_size=512):
        super().__init__()

        self.num_graph_steps = num_graph_steps
        self.num_proposals = 128 if mode == "gt" else 256
        self.mode = mode

        if use_relation: assert use_topdown # only enable use_relation in topdown captioning module

        if num_graph_steps > 0:
            from models.graph_module import GraphModule
            self.graph = GraphModule(128, 128, num_graph_steps, self.num_proposals, 128, num_locals, 
                query_mode, graph_mode, return_edge=use_relation, graph_aggr=graph_aggr,
                return_orientation=use_orientation, num_bins=num_bins, return_distance=use_distance)

        from models.caption_module import SceneCaptionModule, TopDownSceneCaptionModule
        if use_topdown:
            self.caption = TopDownSceneCaptionModule(vocabulary, embeddings, emb_size, 128, 
                hidden_size, self.num_proposals, num_locals, query_mode, use_relation, use_oracle=(mode == "gt"))
        else:
            self.caption = SceneCaptionModule(vocabulary, embeddings, emb_size, 128, hidden_size, self.num_proposals)

    def forward(self, data_dict, use_tf=True, is_eval=False):
        """ Forward pass of the network

        Returns:
            data_dict: dict
        """

        if self.num_graph_steps > 0: data_dict = self.graph(data_dict)

        if self.mode == "gt":
            data_dict = self.caption(data_dict, use_tf, is_eval)
        else:
            data_dict = self.caption(data_dict, use_tf, is_eval)

        return data_dict
