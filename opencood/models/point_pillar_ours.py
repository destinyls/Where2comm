# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from numpy import record
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.dcn_net import DCNNet
from opencood.models.fuse_modules.fusion_net import Where2comm
import torch
import cv2
import time


class PointPillar(nn.Module):
    def __init__(self, args, backbone_fix=False):
        super(PointPillar, self).__init__()
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False
        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        self.dcn = False
        if 'dcn' in args:
            self.dcn = True
            self.dcn_net = DCNNet(args['dcn'])
        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        if backbone_fix:
            self.backbone_fix()
            
    def forward(self, batch_dict):
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        # N, C, H', W'. [N, 384, 100, 352]
        spatial_features = batch_dict['spatial_features']
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        # dcn
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)
        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)
        return psm, rm, spatial_features, spatial_features_2d
    
    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False
        for p in self.scatter.parameters():
            p.requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False
        if self.dcn:
            for p in self.dcn_net.parameters():
                p.requires_grad = False
        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

class PointPillarOurs(nn.Module):
    def __init__(self, args):
        super(PointPillarOurs, self).__init__()
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
        self.compression = False
        if args['compression'] > 0:
            self.compression = True
        self.dcn = False
        if 'dcn' in args:
            self.dcn = True

        self.model_infra = PointPillar(args, args['infra_fix'])
        self.model_vehicle = PointPillar(args, args['vehicle_fix'])
        # self.fusion_net = TransformerFusion(args['fusion_args'])
        self.fusion_net = Where2comm(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']
        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'], kernel_size=1)

        if args['fusion_fix']:
            self.fusion_fix()

    def fusion_fix(self):
        for p in self.fusion_net.parameters():
            p.requires_grad = False
        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def split_data(self, voxel_features, voxel_coords, voxel_num_points, record_len):
        batch_size = voxel_coords[:, 0].max().int().item() + 1
        voxel_features_v, voxel_coords_v, num_points_v = [], [], []
        voxel_features_i, voxel_coords_i, num_points_i = [], [], []        
        for batch_idx in range(batch_size):
            batch_mask = voxel_coords[:, 0] == batch_idx
            voxel_coords[batch_mask, 0] = batch_idx // 2
            if batch_idx % 2 == 0:
                voxel_features_v.append(voxel_features[batch_mask, :])
                voxel_coords_v.append(voxel_coords[batch_mask, :])
                num_points_v.append(voxel_num_points[batch_mask])
            else:
                voxel_features_i.append(voxel_features[batch_mask, :])
                voxel_coords_i.append(voxel_coords[batch_mask, :])
                num_points_i.append(voxel_num_points[batch_mask])

        voxel_features_v = torch.cat(voxel_features_v, 0)
        voxel_coords_v = torch.cat(voxel_coords_v, 0)
        num_points_v = torch.cat(num_points_v, 0)
        voxel_features_i = torch.cat(voxel_features_i, 0)
        voxel_coords_i = torch.cat(voxel_coords_i, 0)
        num_points_i = torch.cat(num_points_i, 0)
        
        batch_dict_v = {'voxel_features': voxel_features_v,
                        'voxel_coords': voxel_coords_v,
                        'voxel_num_points': num_points_v,
                        'record_len': record_len}
        batch_dict_i = {'voxel_features': voxel_features_i,
                        'voxel_coords': voxel_coords_i,
                        'voxel_num_points': num_points_i,
                        'record_len': record_len}
        return batch_dict_v, batch_dict_i

    def forward(self, data_dict, dataset):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        anchor_box = data_dict['anchor_box']

        batch_dict_v, batch_dict_i = self.split_data(voxel_features, voxel_coords, voxel_num_points, record_len)
        psm_single_v, rm_single_v, spatial_features_v, spatial_features_2d_v = self.model_vehicle(batch_dict_v)
        psm_single_i, rm_single_i, spatial_features_i, spatial_features_2d_i= self.model_infra(batch_dict_i)
        
        middle_output_dict_list = []
        middle_data_dict_list = []
        batch_size = voxel_coords[:, 0].max().int().item() + 1
        for i in range(batch_size):
            middle_output_dict = {'psm_single_v': psm_single_v[i],
                                'rm_single_v': rm_single_v[i],
                                'psm_single_i': psm_single_i[i],
                                'rm_single_i': rm_single_i[i],
                                'spatial_features_v': spatial_features_v[i],
                                'spatial_features_i': spatial_features_i[i],
                                'spatial_features_2d_v': spatial_features_2d_v[i],
                                'spatial_features_2d_i': spatial_features_2d_i[i]
                                }
            middle_output_dict_list.append(middle_output_dict)
            
            # 因为路端也在自己的坐标系下做后处理，所以是单位矩阵
            middle_data_dict = {'transformation_matrix': pairwise_t_matrix[i][0,0],
                                'transformation_matrix_10': pairwise_t_matrix[i][0,0], # pairwise_t_matrix[i][1,0]
                                'anchor_box': anchor_box[i]
                                }
            middle_data_dict_list.append(middle_data_dict)

        
        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(middle_output_dict,
                                            record_len,
                                            pairwise_t_matrix, 
                                            self.model_vehicle.backbone,
                                            [self.model_vehicle.shrink_conv, self.cls_head, self.reg_head])
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.model_vehicle.shrink_conv(fused_feature)
        else:
            fused_feature = self.fusion_net(dataset, middle_data_dict_list,middle_output_dict_list,
                                            record_len,
                                            pairwise_t_matrix)
            
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        output_dict = {'psm': psm,
                       'rm': rm,
                       'psm_single_v': psm_single_v,
                       'psm_single_i': psm_single_i,
                       'rm_single_v': rm_single_v,
                       'rm_single_i': rm_single_i,
                       'comm_rate': 0
                       }
        return output_dict
