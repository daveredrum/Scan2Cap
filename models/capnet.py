import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.graph_module import GraphModule
from models.caption_module import SceneCaptionModule, TopDownSceneCaptionModule


class CapNet(nn.Module):
    def __init__(self, num_class, vocabulary, embeddings, num_heading_bin, num_size_cluster, mean_size_arr, 
    input_feature_dim=0, num_proposal=256, num_locals=-1, vote_factor=1, sampling="vote_fps",
    no_caption=False, use_topdown=False, query_mode="corner", 
    graph_mode="graph_conv", num_graph_steps=0, use_relation=False, graph_aggr="add",
    use_orientation=False, num_bins=6, use_distance=False, use_new=False, 
    emb_size=300, hidden_size=512):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.no_caption = no_caption
        self.num_graph_steps = num_graph_steps

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)

        if use_relation: assert use_topdown # only enable use_relation in topdown captioning module

        if num_graph_steps > 0:
            self.graph = GraphModule(128, 128, num_graph_steps, num_proposal, 128, num_locals, 
                query_mode, graph_mode, return_edge=use_relation, graph_aggr=graph_aggr, 
                return_orientation=use_orientation, num_bins=num_bins, return_distance=use_distance)

        # Caption generation
        if not no_caption:
            if use_topdown:
                self.caption = TopDownSceneCaptionModule(vocabulary, embeddings, emb_size, 128, 
                    hidden_size, num_proposal, num_locals, query_mode, use_relation)
            else:
                self.caption = SceneCaptionModule(vocabulary, embeddings, emb_size, 128, hidden_size, num_proposal)

    def forward(self, data_dict, use_tf=True, is_eval=False):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        data_dict = self.backbone_net(data_dict)
                
        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal(xyz, features, data_dict)

        #######################################
        #                                     #
        #           GRAPH ENHANCEMENT         #
        #                                     #
        #######################################

        if self.num_graph_steps > 0: data_dict = self.graph(data_dict)

        #######################################
        #                                     #
        #            CAPTION BRANCH           #
        #                                     #
        #######################################

        # --------- CAPTION GENERATION ---------
        if not self.no_caption:
            data_dict = self.caption(data_dict, use_tf, is_eval)

        return data_dict
