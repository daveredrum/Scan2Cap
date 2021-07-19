import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
# from models.backbone_module import Pointnet2Backbone
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from models.voting_module import VotingModule

class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.input_feature_dim = input_feature_dim

        # --------- 4 SET ABSTRACTION LAYERS ---------
        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, data_dict):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            data_dict: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        
        pointcloud = data_dict["point_clouds"]

        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        data_dict['sa1_inds'] = fps_inds
        data_dict['sa1_xyz'] = xyz
        data_dict['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        data_dict['sa2_inds'] = fps_inds
        data_dict['sa2_xyz'] = xyz
        data_dict['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        data_dict['sa3_inds'] = fps_inds
        data_dict['sa3_xyz'] = xyz
        data_dict['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        data_dict['sa4_inds'] = fps_inds
        data_dict['sa4_xyz'] = xyz
        data_dict['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(data_dict['sa3_xyz'], data_dict['sa4_xyz'], data_dict['sa3_features'], data_dict['sa4_features'])
        features = self.fp2(data_dict['sa2_xyz'], data_dict['sa3_xyz'], data_dict['sa2_features'], features)
        data_dict['fp2_features'] = features
        data_dict['fp2_xyz'] = data_dict['sa2_xyz']
        num_seed = data_dict['fp2_xyz'].shape[1]
        data_dict['fp2_inds'] = data_dict['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds

        return data_dict

class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
            npoint=self.num_proposal,
            # radius=1.6,
            radius=5,
            nsample=512,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )
            
        # Object proposal/detection
        # center coordinates (3),
        # size class+residual(num_size_cluster*4)
        self.proposal = nn.Sequential(
            nn.Conv1d(128,128,1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,3+num_size_cluster*4+self.num_class,1)
        )

    def forward(self, xyz, features, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        # Farthest point sampling (FPS) on votes
        xyz, features, fps_inds = self.vote_aggregation(xyz, features)
        
        # # max pool
        # xyz, features, fps_inds = data_dict['fp2_xyz'], data_dict['fp2_features'], data_dict['fp2_inds']
        # features = features.max(dim=-1, keepdim=True)[0]

        sample_inds = fps_inds

        data_dict['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        data_dict['aggregated_vote_features'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, 128)
        data_dict['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = self.proposal(features)
        data_dict = self.decode_scores(net, data_dict, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)

        return data_dict

    def decode_scores(self, net, data_dict, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
        """
        decode the predicted parameters for the bounding boxes

        """
        net_transposed = net.transpose(2,1).contiguous() # (batch_size, 1024, ..)
        batch_size = net_transposed.shape[0]
        num_proposal = net_transposed.shape[1]

        base_xyz = data_dict['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
        center = base_xyz + net_transposed[:,:,0:3] # (batch_size, num_proposal, 3)
        
        size_scores = net_transposed[:,:,3:3+num_size_cluster]
        size_residuals_normalized = net_transposed[:,:,3+num_size_cluster:3+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
        
        sem_cls_scores = net_transposed[:,:,3+num_size_cluster*4:] # Bxnum_proposalx10

        # store
        data_dict['center'] = center
        data_dict['size_scores'] = size_scores
        data_dict['size_residuals_normalized'] = size_residuals_normalized
        data_dict['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        data_dict['sem_cls_scores'] = sem_cls_scores

        return data_dict


class MaskVoteNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
    input_feature_dim=0, num_proposal=1, vote_factor=1, sampling="vote_fps"):
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

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)

    def forward(self, data_dict):
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
        # data_dict = self.proposal(data_dict)

        return data_dict
