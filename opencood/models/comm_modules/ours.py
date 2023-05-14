# -*- coding: utf-8 -*-
# Author: Yue Hu <phyllis1sjtu@outlook.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        
        self.thre = args['thre']
        self.voxel_size = args['voxel_size']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, pred_box_infra, pred_score_infra, infra_features):

        B, C, H, W = infra_features.shape
        
        # TODO 确认pred_score_infra
        pred_box_infra = pred_box_infra[torch.where(pred_score_infra > self.thre)]
        N = min(pred_box_infra.shape[0], 100)
        pred_box_infra = pred_box_infra[:N,:,:]
        l_corner, _ = torch.min(pred_box_infra, dim=1)
        r_corner, _ = torch.max(pred_box_infra, dim=1)
        center_points_3d = (l_corner + r_corner) / 2
        
        # left and right corner points [[l_x, l_y], [r_x, r_y]]
        corner_points_bev = torch.zeros([center_points_3d.shape[0], 2, 2], dtype=center_points_3d.dtype)
        corner_points_bev[:,0,0] = (l_corner[:,0] + W/2) / self.voxel_size[0]
        corner_points_bev[:,0,1] = (l_corner[:,1] + H/2) / self.voxel_size[1]
        corner_points_bev[:,1,0] = (r_corner[:,0] + W/2) / self.voxel_size[0]
        corner_points_bev[:,1,1] = (r_corner[:,1] + H/2) / self.voxel_size[1]
        bev_size = (corner_points_bev[:,1,1] - corner_points_bev[:,0,1]) * \
                            (corner_points_bev[:,1,0] - corner_points_bev[:,0,0])
        
        # center points
        center_points_bev = torch.zeros([center_points_3d.shape[0], 2], dtype=center_points_3d.dtype)
        center_points_bev[:,0] = (center_points_3d[:,0] + W/2) / self.voxel_size[0]
        center_points_bev[:,1] = (center_points_3d[:,1] + H/2) / self.voxel_size[1]
        center_points_bev_unsqueeze = center_points_bev.unsqueeze(0).to(infra_features.device)
        
        center_points_features = F.grid_sample(infra_features, center_points_bev_unsqueeze.unsqueeze(0), align_corners=False)
        
        Y, X = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing="ij") 
        gaussian_maps_list = []
        for i in range(N):
            gaussian_map = ((X - center_points_bev[i][0])**2 + (Y - center_points_bev[i][1])**2) / (2*bev_size[i]**2)
            gaussian_maps_list.append(gaussian_map)
        gaussian_maps = torch.stack(gaussian_maps_list, dim=0).unsqueeze(0).to(infra_features.device)  #[1, N, H, W]
        center_points_features = center_points_features.transpose(0, 1).transpose(1, 3).expand(C, N, H, W)  # [1, N, 1, C] -> [C, N, H, W]
        select_features = (torch.sum(center_points_features * gaussian_maps, dim=1).unsqueeze(0)) / N
        
        return select_features