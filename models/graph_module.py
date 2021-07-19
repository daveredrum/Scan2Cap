import torch

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as GUtils

from torch_geometric.data import Data as GData
from torch_geometric.data import DataLoader as GDataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.typing import Adj, Size

from scipy.sparse import coo_matrix

import os
import sys
sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from utils.box_util import box3d_iou_batch_tensor
from lib.config import CONF

class EdgeConv(MessagePassing):
    def __init__(self, in_size, out_size, aggregation="add"):
        super().__init__(aggr=aggregation)
        self.in_size = in_size
        self.out_size = out_size

        self.map_edge = nn.Sequential(
            nn.Linear(2 * in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size)
        )
        # self.map_node = nn.Sequential(
        #     nn.Linear(out_size, out_size),
        #     nn.ReLU()
        # )

    def forward(self, x, edge_index):
        # x has shape [N, in_size]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            adj (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                        kwargs)

        msg_kwargs = self.inspector.distribute('message', coll_dict)
        message = self.message(**msg_kwargs)

        # For `GNNExplainer`, we require a separate message and aggregate
        # procedure since this allows us to inject the `edge_mask` into the
        # message passing computation scheme.
        if self.__explain__:
            edge_mask = self.__edge_mask__.sigmoid()
            # Some ops add self-loops to `edge_index`. We need to do the
            # same for `edge_mask` (but do not train those).
            if message.size(self.node_dim) != edge_mask.size(0):
                loop = edge_mask.new_ones(size[0])
                edge_mask = torch.cat([edge_mask, loop], dim=0)
            assert message.size(self.node_dim) == edge_mask.size(0)
            message = message * edge_mask.view([-1] + [1] * (message.dim() - 1))

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        out = self.aggregate(message, **aggr_kwargs)

        update_kwargs = self.inspector.distribute('update', coll_dict)
        
        return self.update(out, **update_kwargs), message

    def message(self, x_i, x_j):
        # x_i has shape [E, in_size]
        # x_j has shape [E, in_size]

        edge = torch.cat([x_i, x_j - x_i], dim=1)  # edge has shape [E, 2 * in_size]
        # edge = torch.cat([x_i, x_j], dim=1)  # edge has shape [E, 2 * in_size]
        
        return self.map_edge(edge)

    def update(self, x_i):
        # x has shape [N, out_size]

        # return self.map_node(x_i)
        return x_i

class GraphModule(nn.Module):
    def __init__(self, in_size, out_size, num_layers, num_proposals, feat_size, num_locals, 
        query_mode="corner", graph_mode="graph_conv", return_edge=False, graph_aggr="add", 
        return_orientation=False, num_bins=6, return_distance=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.num_proposals = num_proposals
        self.feat_size = feat_size
        
        self.num_locals = num_locals
        self.query_mode = query_mode

        # graph layers
        self.graph_mode = graph_mode
        self.gc_layers = nn.ModuleList()
        for _ in range(num_layers):
            if graph_mode == "graph_conv":
                self.gc_layers.append(GCNConv(in_size, out_size))
            elif graph_mode == "edge_conv":
                self.gc_layers.append(EdgeConv(in_size, out_size, graph_aggr))
            else:
                raise ValueError("invalid graph mode, choices: [\"graph_conv\", \"edge_conv\"]")

        # graph edges
        self.return_edge = return_edge
        self.return_orientation = return_orientation
        self.return_distance = return_distance
        self.num_bins = num_bins

        # output final edges
        if self.return_orientation: 
            assert self.graph_mode == "edge_conv"
            self.edge_layer = EdgeConv(in_size, out_size, graph_aggr)
            self.edge_predict = nn.Linear(out_size, num_bins + 1)

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

    def _create_adjacent_mat(self, data_dict, object_masks):
        batch_size, num_objects = object_masks.shape
        adjacent_mat = torch.zeros(batch_size, num_objects, num_objects).cuda()

        for obj_id in range(num_objects):
            target_ids = torch.LongTensor([obj_id for _ in range(batch_size)]).cuda()
            adjacent_entry = self._query_locals(data_dict, target_ids, object_masks, include_self=False) # batch_size, num_objects
            adjacent_mat[:, obj_id] = adjacent_entry

        return adjacent_mat

    def _feed(self, graph):
        feat, edge = graph.x, graph.edge_index

        for layer in self.gc_layers:
            if self.graph_mode == "graph_conv":
                feat = layer(feat, edge)
                message = None
            elif self.graph_mode == "edge_conv":
                feat, message = layer(feat, edge)

        return feat, message

    def forward(self, data_dict):
        obj_feats = data_dict["bbox_feature"] # batch_size, num_proposals, feat_size
        object_masks = data_dict["bbox_mask"] # batch_size, num_proposals

        batch_size, num_objects, _ = obj_feats.shape
        adjacent_mat = self._create_adjacent_mat(data_dict, object_masks) # batch_size, num_proposals, num_proposals

        new_obj_feats = torch.zeros(batch_size, num_objects, self.feat_size).cuda()
        edge_indices = torch.zeros(batch_size, 2, num_objects * self.num_locals).cuda()
        edge_feats = torch.zeros(batch_size, num_objects, self.num_locals, self.out_size).cuda()
        edge_preds = torch.zeros(batch_size, num_objects * self.num_locals, self.num_bins+1).cuda()
        num_sources = torch.zeros(batch_size).long().cuda()
        num_targets = torch.zeros(batch_size).long().cuda()
        for batch_id in range(batch_size):
            # valid object masks
            batch_object_masks = object_masks[batch_id] # num_objects

            # create adjacent matric for this scene
            batch_adjacent_mat = adjacent_mat[batch_id] # num_objects, num_objects
            batch_adjacent_mat = batch_adjacent_mat[batch_object_masks == 1, :][:, batch_object_masks == 1] # num_valid_objects, num_valid_objects

            # initialize graph for this scene
            sparse_mat = coo_matrix(batch_adjacent_mat.detach().cpu().numpy())
            batch_edge_index, edge_attr = GUtils.from_scipy_sparse_matrix(sparse_mat)
            batch_obj_feats = obj_feats[batch_id, batch_object_masks == 1] # num_valid_objects, in_size
            batch_graph = GData(x=batch_obj_feats, edge_index=batch_edge_index.cuda())

            # graph conv
            node_feat, edge_feat = self._feed(batch_graph)

            # # skip connection
            # node_feat += batch_obj_feats
            # new_obj_feats[batch_id, batch_object_masks == 1] = node_feat

            # output last edge
            if self.return_orientation:
                # output edge
                try:
                    num_src_objects = len(set(batch_edge_index[0].cpu().numpy()))
                    num_tar_objects = int(edge_feat.shape[0] / num_src_objects)

                    num_sources[batch_id] = num_src_objects
                    num_targets[batch_id] = num_tar_objects

                    edge_feat = edge_feat[:num_src_objects*num_tar_objects] # in case there are less than 10 neighbors                    
                    edge_feats[batch_id, :num_src_objects, :num_tar_objects] = edge_feat.view(num_src_objects, num_tar_objects, self.out_size)
                    edge_indices[batch_id, :, :num_src_objects*num_tar_objects] = batch_edge_index[:, :num_src_objects*num_tar_objects]
                    
                    _, edge_feat = self.edge_layer(node_feat, batch_edge_index.cuda())
                    edge_pred = self.edge_predict(edge_feat)
                    edge_preds[batch_id, :num_src_objects*num_tar_objects] = edge_pred

                except Exception:
                    print("error occurs when dealing with graph, skipping...")

            # skip connection
            batch_obj_feats += node_feat
            new_obj_feats[batch_id, batch_object_masks == 1] = batch_obj_feats

        # store
        data_dict["bbox_feature"] = new_obj_feats
        data_dict["adjacent_mat"] = adjacent_mat
        data_dict["edge_index"] = edge_indices
        data_dict["edge_feature"] = edge_feats
        data_dict["num_edge_source"] = num_sources
        data_dict["num_edge_target"] = num_targets
        data_dict["edge_orientations"] = edge_preds[:, :, :-1]
        data_dict["edge_distances"] = edge_preds[:, :, -1]

        return data_dict