import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd())) # HACK add the lib folder
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes

class PointnetEncoder(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
       
       Input should be point clouds for scenes with the last digit 
       as a binary mask for indicating the points on target object
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0, num_classes=18, whole_scene=False):
        super().__init__()

        self.input_feature_dim = input_feature_dim
        self.num_classes = num_classes
        self.whole_scene = whole_scene

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

        self.map = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, num_classes)

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def _chunk_pc(self, pc, chunk_size):
        num_valid_objects = pc.shape[0]
        num_chunks = int(np.ceil(num_valid_objects / chunk_size))

        chunks = torch.chunk(pc, num_chunks, dim=0)

        return chunks

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

        if self.whole_scene:
            object_masks = data_dict["target_masks"] # batch_size, num_bboxes

            batch_size, num_bboxes, _, _ = pointcloud.shape
            enc_features = torch.zeros(batch_size, num_bboxes, 128).cuda()
            enc_preds = torch.zeros(batch_size, num_bboxes, self.num_classes).cuda()
            for i in range(batch_size):
                current_mask = object_masks[i] == 1
                num_valid_objects = current_mask.sum()
                pc = pointcloud[i, current_mask] # num_valid_objects, num_points, num_feat

                chunk_size = batch_size
                chunks = self._chunk_pc(pc, chunk_size)

                offset = 0
                for chunk_id, chunk in enumerate(chunks):
                    actual_chunk_size = chunk.shape[0]
                    xyz, features = self._break_up_pc(chunk)

                    # --------- 4 SET ABSTRACTION LAYERS ---------
                    xyz, features, fps_inds = self.sa1(xyz, features)
                    data_dict["sa1_inds"] = fps_inds
                    data_dict["sa1_xyz"] = xyz
                    data_dict["sa1_features"] = features

                    xyz, features, fps_inds = self.sa2(xyz, features)
                    data_dict["sa2_inds"] = fps_inds
                    data_dict["sa2_xyz"] = xyz
                    data_dict["sa2_features"] = features

                    xyz, features, fps_inds = self.sa3(xyz, features)
                    data_dict["sa3_inds"] = fps_inds
                    data_dict["sa3_xyz"] = xyz
                    data_dict["sa3_features"] = features

                    xyz, features, fps_inds = self.sa4(xyz, features)
                    data_dict["sa4_inds"] = fps_inds
                    data_dict["sa4_xyz"] = xyz
                    data_dict["sa4_features"] = features


                    features = features.max(-1)[0] # max pool
                    features = self.map(features)
                    preds = self.classifier(features)

                    enc_features[i, offset: offset + actual_chunk_size] = features
                    enc_preds[i, offset: offset + actual_chunk_size] = preds

                    offset += actual_chunk_size

            data_dict["enc_features"] = enc_features
            data_dict["enc_preds"] = enc_preds         
            pass
        else:
            xyz, features = self._break_up_pc(pointcloud)

            # --------- 4 SET ABSTRACTION LAYERS ---------
            xyz, features, fps_inds = self.sa1(xyz, features)
            data_dict["sa1_inds"] = fps_inds
            data_dict["sa1_xyz"] = xyz
            data_dict["sa1_features"] = features

            xyz, features, fps_inds = self.sa2(xyz, features)
            data_dict["sa2_inds"] = fps_inds
            data_dict["sa2_xyz"] = xyz
            data_dict["sa2_features"] = features

            xyz, features, fps_inds = self.sa3(xyz, features)
            data_dict["sa3_inds"] = fps_inds
            data_dict["sa3_xyz"] = xyz
            data_dict["sa3_features"] = features

            xyz, features, fps_inds = self.sa4(xyz, features)
            data_dict["sa4_inds"] = fps_inds
            data_dict["sa4_xyz"] = xyz
            data_dict["sa4_features"] = features

            features = features.max(-1)[0] # max pool
            features = self.map(features)
            preds = self.classifier(features)

            data_dict["enc_features"] = features
            data_dict["enc_preds"] = preds
        
        return data_dict

if __name__=="__main__":
    encoder = PointnetEncoder(input_feature_dim=3 + 1).cuda()
    encoder.eval()
    inputs = {"point_clouds": torch.rand(16,20000,6 + 1).cuda()}
    out = encoder(inputs)
    for key in sorted(out.keys()):
        print(key, "\t", out[key].shape)