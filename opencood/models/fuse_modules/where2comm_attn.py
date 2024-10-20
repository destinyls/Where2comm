# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Implementation of V2VNet Fusion
"""

from turtle import update
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.comm_modules.where2comm import Communication

from opencood.visualization import simple_vis
from opencood.models.fuse_modules.gaussian import Gaussian
from opencood.models.fuse_modules.models_mae import mae_vit_custom_patch1
from opencood.models.fuse_modules.how2comm_preprocess import How2commPreprocess
from opencood.models.fuse_modules.GlobalAlign import GlobalAlign

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context

class AttenFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttenFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x

class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]

class EncodeLayer(nn.Module):
    def __init__(self, channels, n_head=8, dropout=0):
        super(EncodeLayer, self).__init__()
        self.attn = nn.MultiheadAttention(channels, n_head, dropout)
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, q, k, v, confidence_map=None):
        """
        order (seq, batch, feature)
        Args:
            q: (1, H*W, C)
            k: (N, H*W, C)
            v: (N, H*W, C)
        Returns:
            outputs: ()
        """
        residual = q
        if confidence_map is not None:
            context, weight = self.attn(q,k,v, quality_map=confidence_map) # (1, H*W, C)
        else:
            context, weight = self.attn(q,k,v) # (1, H*W, C)
        context = self.dropout1(context)
        output1 = self.norm1(residual + context)

        # feed forward net
        residual = output1 # (1, H*W, C)
        context = self.linear2(self.relu(self.linear1(output1)))
        context = self.dropout2(context)
        output2 = self.norm2(residual + context)

        return output2

class TransformerFusion(nn.Module):
    def __init__(self, channels=256, n_head=8, with_spe=True, with_scm=True, dropout=0):
        super(TransformerFusion, self).__init__()

        self.encode_layer = EncodeLayer(channels, n_head, dropout)
        self.with_spe = with_spe
        self.with_scm = with_scm
        
    def forward(self, batch_neighbor_feature, batch_neighbor_feature_pe, batch_confidence_map, record_len):
        x_fuse = []
        B = len(record_len)
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            neighbor_feature = batch_neighbor_feature[b]
            _, C, H, W = neighbor_feature.shape
            neighbor_feature_flat = neighbor_feature.view(N,C,H*W)  # (N, C, H*W)

            if self.with_spe:
                neighbor_feature_pe = batch_neighbor_feature_pe[b]
                neighbor_feature_flat_pe = neighbor_feature_pe.view(N,C,H*W)  # (N, C, H*W)
                query = neighbor_feature_flat_pe[0:1,...].permute(0,2,1)  # (1, H*W, C)
                key = neighbor_feature_flat_pe.permute(0,2,1)  # (N, H*W, C)
            else:
                query = neighbor_feature_flat[0:1,...].permute(0,2,1)  # (1, H*W, C)
                key = neighbor_feature_flat.permute(0,2,1)  # (N, H*W, C)
            
            value = neighbor_feature_flat.permute(0,2,1)

            if self.with_scm:
                confidence_map = batch_confidence_map[b]
                fused_feature = self.encode_layer(query, key, value, confidence_map)  # (1, H*W, C)
            else:
                fused_feature = self.encode_layer(query, key, value)  # (1, H*W, C)
            
            fused_feature = fused_feature.permute(0,2,1).reshape(1, C, H, W)

            x_fuse.append(fused_feature)
        x_fuse = torch.concat(x_fuse, dim=0)
        return x_fuse

def add_pe_map(x):
    # scale = 2 * math.pi
    temperature = 10000
    num_pos_feats = x.shape[-3] // 2  # positional encoding dimension. C = 2d

    mask = torch.zeros([x.shape[-2], x.shape[-1]], dtype=torch.bool, device=x.device)  #[H, W]
    not_mask = ~mask
    y_embed = not_mask.cumsum(0, dtype=torch.float32)  # [H, W]
    x_embed = not_mask.cumsum(1, dtype=torch.float32)  # [H, W]

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)  # [0,1,2,...,d]
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # 10000^(2k/d), k is [0,0,1,1,...,d/2,d/2]

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)  # [C, H, W]

    if len(x.shape) == 4:
        x_pe = x + pos[None,:,:,:]
    elif len(x.shape) == 5:
        x_pe = x + pos[None,None,:,:,:]
    return x_pe

class Where2comm(nn.Module):
    def __init__(self, args):
        super(Where2comm, self).__init__()

        self.communication = False
        self.round = 1
        self.multi_scale_map = {0: [100, 252, 64], 1: [50, 126, 128], 2: [25, 63, 256]} # HWC
        if 'communication' in args:
            self.communication = True
            self.naive_communication = Communication(args['communication'])
            if 'round' in args['communication']:
                self.round = args['communication']['round']
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        self.gaussian = Gaussian(args)
        self.downsample_factor = {100: (20, 36, 2), 50: (10, 18, 1), 25: (5, 9, 1)}
            
        self.mode = args['para']['mode']
        if self.mode == "maskAndRec" or self.mode == "onlyMask":
            self.mask_ratio = args['para']['mask_ratio']
        
        self.his_flag = args['para']['his_flag']
        self.how2comm = How2commPreprocess(args['para'])
        
        if self.multi_scale:  # True
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            self.mae_modules = nn.ModuleList()
            self.globalAlign = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == 'ATTEN':
                    fuse_network = AttenFusion(num_filters[idx])
                elif self.agg_mode == 'MAX':
                    fuse_network = MaxFusion()
                elif self.agg_mode == 'Transformer':
                    fuse_network = TransformerFusion(
                                                channels=num_filters[idx], 
                                                n_head=args['agg_operator']['n_head'], 
                                                with_spe=args['agg_operator']['with_spe'], 
                                                with_scm=args['agg_operator']['with_scm'])
                self.fuse_modules.append(fuse_network)
                
                HWC = self.multi_scale_map[idx]
                min_hw, max_hw = min(HWC[0], HWC[1]), max(HWC[0], HWC[1])

                downsample_factor_h, downsample_factor_w, patch_size = self.downsample_factor[min_hw]
                self.mae_modules.append(mae_vit_custom_patch1(img_size=(downsample_factor_h, downsample_factor_w), patch_size=patch_size, in_chans=HWC[2], norm_pix_loss=False))
                self.globalAlign.append(GlobalAlign(in_channel=num_filters[idx]))
        else:
            if self.agg_mode == 'ATTEN':
                self.fuse_modules = AttenFusion(args['agg_operator']['feature_dim'])
            elif self.agg_mode == 'MAX':
                self.fuse_modules = MaxFusion()   
            elif self.agg_mode == 'Transformer':
                self.fuse_network = TransformerFusion(
                                            channels=args['agg_operator']['feature_dim'], 
                                            n_head=args['agg_operator']['n_head'], 
                                            with_spe=args['agg_operator']['with_spe'], 
                                            with_scm=args['agg_operator']['with_scm'])     

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, rm, record_len, pairwise_t_matrix, dataset, data_dict, output_dict, backbone=None, heads=None, his_features=None, fur_features=None, time=None):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, L, 4, 4) 
            
        Returns
        -------
        Fused feature.
        """        
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        if his_features != [] and fur_features != []: # flow_pre
            cur_feature = torch.stack([fea[1] for fea in self.regroup(x, record_len)])
            infra_predict_feature, loss_offset = self.how2comm(cur_feature, his_features, fur_features, time) 
        else:
            loss_offset = None
              
        if self.multi_scale:   # True
            pred_box_infra_list, pred_score_infra_list = [], []
            
            for b in range(B):
                pred_box_infra, pred_score_infra = dataset.post_process(data_dict[b], output_dict[b], selected_agent=1, middle_post_process=True)  # infra
                pred_box_infra_list.append(pred_box_infra)
                pred_score_infra_list.append(pred_score_infra) 
                
            ups = []
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(x)
                if his_features != [] and fur_features != []:
                    infra_prefea = backbone.resnet(infra_predict_feature)
            
            loss_mae = None
            loss_align = None
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)

                ############ 1. Communication (Mask the features) #########
                if i==0:
                    if self.communication:
                        batch_confidence_maps = self.regroup(rm, record_len)
                        _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                        x = x * communication_masks
                    else:
                        communication_rates = torch.tensor(0).to(x.device)  # False
                
                ############ 2. Split the confidence map #######################
                # split x:[(L1, C, H, W), (L2, C, H, W), ...]
                # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
                batch_node_features = self.regroup(x, record_len)
                
                ############ 3. Fusion ####################################
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    # (N,N,4,4)   t_matrix[i, j]-> from i to j
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    if self.mode == "baseline":
                        neighbor_feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H, W))  
                        fuse_feature = self.fuse_modules[i](neighbor_feature)   # [2, 64, 100, 252]
                    elif self.mode == "onlyVehFea":
                        fuse_feature = node_features[0] # torch.Size([64, 100, 252])
                    elif self.mode == "maskAndRec":   
                        pred_box_infra, pred_score_infra = pred_box_infra_list[b], pred_score_infra_list[b]
                        gaussian_maps = self.gaussian(pred_box_infra, torch.zeros_like(node_features[1].unsqueeze(0)), i)                 
                        
                        infra_features = node_features[1].unsqueeze(0) * (gaussian_maps > 0).float()  
                        n, c, h, w = infra_features.shape[0], infra_features.shape[1], infra_features.shape[2], infra_features.shape[3]
 
                        ''' mae restruction '''
                        HWC = self.multi_scale_map[i]
                        max_hw, min_hw = max(HWC[0], HWC[1]), min(HWC[0], HWC[1])
                                            
                        downsample_factor_h, downsample_factor_w, _ = self.downsample_factor[min_hw]
                        # 调整 infra_feature形状 用于 mae
                        infra_features = infra_features.view(infra_features.shape[1], infra_features.shape[2] // downsample_factor_h, downsample_factor_h, infra_features.shape[3] // downsample_factor_w, downsample_factor_w)
                        infra_features = infra_features.permute(0, 1, 3, 2, 4).contiguous()
                        infra_features = infra_features.view(infra_features.shape[0], -1, downsample_factor_h, downsample_factor_w)
                        infra_features = infra_features.permute(1, 0, 2, 3).contiguous()

                        pred, mask = self.mae_modules[i](infra_features, mask_ratio=self.mask_ratio)  # random masked and reconstruction
                        hw = self.mae_modules[i].get_hw(infra_features)
                        mask = self.mae_modules[i].unpatchify(mask.unsqueeze(-1).repeat(1, 1, int(self.mae_modules[i].patch_embed.patch_size[0])**2), hw)                    
                        mask = self.mae_modules[i].patchify(mask)[:, :, 0]
                        
                        if self.training:   # loss_mae 累加
                            if loss_mae is None:
                                mask[:, :] = 1
                                loss_mae = self.mae_modules[i].forward_loss(infra_features, pred, mask)
                            else:
                                mask[:, :] = 1
                                loss_mae += self.mae_modules[i].forward_loss(infra_features, pred, mask)   
                        
                        infra_features_mae = self.mae_modules[i].unpatchify(pred, hw)[:, :, :min_hw, :]

                        infra_features_mae = infra_features_mae.permute(1, 0, 2, 3).contiguous()
                        infra_features_mae = infra_features_mae.view(c, h // downsample_factor_h, w // downsample_factor_w, downsample_factor_h, downsample_factor_w)
                        infra_features_mae = infra_features_mae.permute(0, 1, 3, 2, 4).contiguous()
                        infra_features_mae = infra_features_mae.view(1, c, h, w)

                        node_features = torch.cat((node_features[0].unsqueeze(0), infra_features_mae), dim=0)   # vehicle+infra [2, 64, 100, 252]
                        neighbor_feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H, W))
                        fuse_feature = self.fuse_modules[i](neighbor_feature) 
                        fuse_feature = torch.cat((fuse_feature.unsqueeze(0), gaussian_maps), dim=0) 
                        fuse_feature = warp_affine_simple(fuse_feature, t_matrix[0, :, :, :], (H, W))
                        fuse_feature = self.fuse_modules[i](fuse_feature)
                    elif self.mode == "onlyMask":   
                        pred_box_infra, pred_score_infra = pred_box_infra_list[b], pred_score_infra_list[b]
                        gaussian_maps = self.gaussian(pred_box_infra, torch.zeros_like(node_features[1].unsqueeze(0)), i)                 
                        
                        infra_features = node_features[1].unsqueeze(0) * (gaussian_maps > 0).float()  
                        n, c, h, w = infra_features.shape[0], infra_features.shape[1], infra_features.shape[2], infra_features.shape[3]
 
                        HWC = self.multi_scale_map[i]
                        max_hw, min_hw = max(HWC[0], HWC[1]), min(HWC[0], HWC[1])
                                            
                        downsample_factor_h, downsample_factor_w, _ = self.downsample_factor[min_hw]

                        infra_features = infra_features.view(infra_features.shape[1], infra_features.shape[2] // downsample_factor_h, downsample_factor_h, infra_features.shape[3] // downsample_factor_w, downsample_factor_w)
                        infra_features = infra_features.permute(0, 1, 3, 2, 4).contiguous()
                        infra_features = infra_features.view(infra_features.shape[0], -1, downsample_factor_h, downsample_factor_w)
                        infra_features = infra_features.permute(1, 0, 2, 3).contiguous()

                        pred, mask = self.mae_modules[i].only_mask(infra_features, mask_ratio=self.mask_ratio)  # random mask
                        hw = self.mae_modules[i].get_hw(infra_features)
                        mask = self.mae_modules[i].unpatchify(mask.unsqueeze(-1).repeat(1, 1, int(self.mae_modules[i].patch_embed.patch_size[0])**2), hw)                    
                        mask = self.mae_modules[i].patchify(mask)[:, :, 0]
                        
                        infra_features_mae = self.mae_modules[i].unpatchify(pred, hw)[:, :, :min_hw, :]

                        infra_features_mae = infra_features_mae.permute(1, 0, 2, 3).contiguous()
                        infra_features_mae = infra_features_mae.view(c, h // downsample_factor_h, w // downsample_factor_w, downsample_factor_h, downsample_factor_w)
                        infra_features_mae = infra_features_mae.permute(0, 1, 3, 2, 4).contiguous()
                        infra_features_mae = infra_features_mae.view(1, c, h, w)

                        node_features = torch.cat((node_features[0].unsqueeze(0), infra_features_mae), dim=0)   # vehicle+infra [2, 64, 100, 252]
                        neighbor_feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H, W))  
                        fuse_feature = self.fuse_modules[i](neighbor_feature) 
                    elif self.mode == "flowPre":
                        if not self.training : # inference predict_flow
                            node_features = torch.cat((node_features[0].unsqueeze(0), infra_prefea[i][b].unsqueeze(0)), dim=0)   # vehicle+infra [2, 64, 100, 252]
                        # finetune
                        node_features = torch.cat((node_features[0].unsqueeze(0), infra_prefea[i][b].unsqueeze(0)), dim=0)   # vehicle+infra [2, 64, 100, 252]
                        neighbor_feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H, W))
                        fuse_feature = self.fuse_modules[i](neighbor_feature)   # [2, 64, 100, 252]  
                    elif self.mode == "correctPosition":   # 暂时不考虑这个
                        neighbor_feature = warp_affine_simple(node_features, t_matrix[0, :, :, :], (H, W))
                        vehicle_fea = node_features[0].unsqueeze(0)
                        infra_fea = node_features[1].unsqueeze(0)
                        cor_infra_frea = self.globalAlign[i](infra_fea, vehicle_fea)
                        
                        if self.training:   # loss_align 累加
                            if loss_align is None:
                                loss_align = self.globalAlign[i].calculate_loss(cor_infra_frea, infra_fea)
                            else:
                                loss_align += self.globalAlign[i].calculate_loss(cor_infra_frea, infra_fea)
                                
                        fuse_feature = torch.cat((cor_infra_frea, vehicle_fea), dim=0)
                        fuse_feature = warp_affine_simple(fuse_feature, t_matrix[0, :, :, :], (H, W))
                        fuse_feature = self.fuse_modules[i](fuse_feature)
                                      
                    x_fuse.append(fuse_feature)
                x_fuse = torch.stack(x_fuse)

                ############ 4. Deconv ####################################
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)
                
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
        else:
            ############ 1. Split the features #######################
            # split x:[(L1, C, H, W), (L2, C, H, W), ...]
            # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
            batch_node_features = self.regroup(x, record_len)
            batch_confidence_maps = self.regroup(rm, record_len)

            ############ 2. Communication (Mask the features) #########
            if self.communication:
                _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
            else:
                communication_rates = torch.tensor(0).to(x.device)
            
            ############ 3. Fusion ####################################
            x_fuse = []
            for b in range(B):
                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                node_features = batch_node_features[b]
                if self.communication:
                    node_features = node_features * communication_masks[b]
                neighbor_feature = warp_affine_simple(node_features,
                                                t_matrix[0, :, :, :],
                                                (H, W))
                x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)
        
        return x_fuse, communication_rates, {}, loss_mae, loss_offset, loss_align
    
