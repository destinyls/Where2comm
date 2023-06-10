# -*- coding: utf-8 -*-
# Author: Yue Hu <phyllis1sjtu@outlook.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import RoIAlign

class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        
        self.voxel_size = args['voxel_size']
        self.lidar_range = args['lidar_range']
        self.downsample_rate = args['downsample_rate']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        self.roi_align = RoIAlign(output_size=[1,1], spatial_scale=1, sampling_ratio=1)
        
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, pred_box_infra, infra_features):
        _, C, H, W = infra_features.shape
        device = infra_features.device
        
        N = min(pred_box_infra.shape[0], 20)
        assert N > 0
        pred_box_infra = pred_box_infra[:N,:,:]
        
        l_corner, _ = torch.min(pred_box_infra, dim=1)
        r_corner, _ = torch.max(pred_box_infra, dim=1)
        center_points_3d = torch.mean(pred_box_infra, dim=1)

        mask = (center_points_3d[:, 0] > self.lidar_range[0]) & (center_points_3d[:, 0] < self.lidar_range[3])\
           & (center_points_3d[:, 1] > self.lidar_range[1]) & (
                   center_points_3d[:, 1] < self.lidar_range[4]) \
           & (center_points_3d[:, 2] > self.lidar_range[2]) & (
                   center_points_3d[:, 2] < self.lidar_range[5])
        center_points_3d = center_points_3d[mask]
        l_corner = l_corner[mask]
        r_corner = r_corner[mask]
        
        if center_points_3d.shape[0] == 0:
            select_features = torch.zeros([1, C, H, W], dtype=infra_features.dtype).to(device)
            return select_features
        
        # left and right corner points [[l_x, l_y], [r_x, r_y]]
        # u = (x-lidar_x_range_min)/vx + W/2, v = (y-lidar_y_range_min)/vy + H/2
        corner_points_bev = torch.zeros([center_points_3d.shape[0], 2, 2], dtype=center_points_3d.dtype).to(device)
        corner_points_bev[:,0,0] = (l_corner[:,0] - self.lidar_range[0]) / (self.voxel_size[0] * self.downsample_rate)
        corner_points_bev[:,0,1] = (l_corner[:,1] - self.lidar_range[1]) / (self.voxel_size[1] * self.downsample_rate)
        corner_points_bev[:,1,0] = (r_corner[:,0] - self.lidar_range[0]) / (self.voxel_size[0] * self.downsample_rate)
        corner_points_bev[:,1,1] = (r_corner[:,1] - self.lidar_range[1]) / (self.voxel_size[1] * self.downsample_rate)
        bev_size = (corner_points_bev[:,1,1] - corner_points_bev[:,0,1]) * \
                            (corner_points_bev[:,1,0] - corner_points_bev[:,0,0])
        
        # center points
        # center_points_bev = torch.zeros([center_points_3d.shape[0], 2], dtype=center_points_3d.dtype)
        # center_points_bev[:,0] = (center_points_3d[:,0] - self.lidar_range[0]) / (self.voxel_size[0] * self.downsample_rate)
        # center_points_bev[:,1] = (center_points_3d[:,1] - self.lidar_range[1]) / (self.voxel_size[1] * self.downsample_rate)
        # center_points_bev_unsqueeze = center_points_bev.unsqueeze(0).unsqueeze(0).to(device)
        
        # center_points_features = F.grid_sample(infra_features, center_points_bev_unsqueeze, align_corners=True)
        center_points_bev = torch.zeros([center_points_3d.shape[0], 5], dtype=center_points_3d.dtype).to(device)
        center_points_bev[:,1] = (l_corner[:,0] - self.lidar_range[0]) / (self.voxel_size[0] * self.downsample_rate)
        center_points_bev[:,2] = (l_corner[:,1] - self.lidar_range[1]) / (self.voxel_size[1] * self.downsample_rate)
        center_points_bev[:,3] = (r_corner[:,0] - self.lidar_range[0]) / (self.voxel_size[0] * self.downsample_rate)
        center_points_bev[:,4] = (r_corner[:,1] - self.lidar_range[1]) / (self.voxel_size[1] * self.downsample_rate)
        center_points_bev[:,0] = 0 # 只有一个batch
        center_points_features = self.roi_align(infra_features, center_points_bev)
        
        ##############################
        Y, X = torch.meshgrid([torch.arange(H, device=device), torch.arange(W, device=device)], indexing="ij") 
        gaussian_maps_list = []
        for i in range(N):
            gaussian_map = torch.exp((-(X - center_points_bev[i][0])**2 - (Y - center_points_bev[i][1])**2) / (2*bev_size[i]**2))
            gaussian_maps_list.append(gaussian_map)
        gaussian_maps = torch.stack(gaussian_maps_list, dim=0).unsqueeze(0).to(device)  #[1, N, H, W]
        # center_points_features = center_points_features.transpose(0, 1).transpose(1, 3).expand(C, N, H, W)  # [1, N, 1, C] -> [C, N, H, W]
        center_points_features = center_points_features.transpose(0, 1).expand(C, N, H, W)  # [N, C, 1, 1] -> [C, N, H, W]
        select_features = (torch.sum(center_points_features * gaussian_maps, dim=1) / N ).unsqueeze(0)
        
        ###############################
        # Y, X = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing="ij") 
        # gaussian_features_sum = torch.zeros([C, H, W], dtype=center_points_features.dtype).to(device)
        # for i in range(N):
        #     gaussian_map = torch.exp((-(X - center_points_bev[i][0])**2 - (Y - center_points_bev[i][1])**2) / (2*bev_size[i]**2))
        #     gaussian_map = gaussian_map.unsqueeze(0).to(device)
        #     center_points_features_ith = center_points_features[0, :, 0, i].unsqueeze(1).unsqueeze(2).expand(C, H, W)
        #     gaussian_features = gaussian_map * center_points_features_ith
        #     gaussian_features_sum += gaussian_features
        # select_features = gaussian_features_sum.unsqueeze(0) / N
        
        assert not torch.isnan(select_features).any() and not torch.isinf(select_features).any()

        return select_features