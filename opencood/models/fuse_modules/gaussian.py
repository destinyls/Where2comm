import os
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import RoIAlign

class Gaussian(nn.Module):
    def __init__(self, args):
        super(Gaussian, self).__init__()
        self.voxel_size = args['voxel_size']
        self.lidar_range = args['lidar_range']
        self.downsample_rate = args['downsample_rate'] * 2 
        if 'gaussian_smooth' in args:
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

    def forward(self, pred_box_infra, infra_features, level, sample_idx):
        downsample_rate = self.downsample_rate * math.pow(2, level)
        _, C, H, W = infra_features.shape
        device = infra_features.device
        if pred_box_infra is None:
            return torch.zeros_like(infra_features)
        N = min(pred_box_infra.shape[0], 20)
        if N == 0:
            return torch.zeros_like(infra_features)

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
        center_points_3d_bev = torch.zeros([center_points_3d.shape[0], 2], dtype=center_points_3d.dtype).to(device)
        center_points_3d_bev[:, 0] = (center_points_3d[:, 0] - self.lidar_range[0]) / (self.voxel_size[0] * downsample_rate)
        center_points_3d_bev[:, 1] = (center_points_3d[:, 1] - self.lidar_range[1]) / (self.voxel_size[1] * downsample_rate)

        Y, X = torch.meshgrid([torch.arange(H, device=device), torch.arange(W, device=device)], indexing="ij") 
        gaussian_maps_list = []
        for i in range(N):
            gaussian_map = torch.exp((-(X - center_points_3d_bev[i][0])**2 - (Y - center_points_3d_bev[i][1])**2) / (5**2))
            gaussian_maps_list.append(gaussian_map)
        gaussian_maps = torch.stack(gaussian_maps_list, dim=0).unsqueeze(0).to(device)  #[1, N, H, W]
        '''
        gaussian_maps_demo = torch.sum(torch.abs(gaussian_maps), dim=1).detach().cpu().numpy()[0] * 100
        cv2.imwrite(os.path.join("demo", "infra_features_demo_2_" + str(sample_idx.cpu().numpy()) + ".jpg"), gaussian_maps_demo)
        '''

        '''
        center_points_features = center_points_features.transpose(0, 1).expand(C, N, H, W)  # [N, C, 1, 1] -> [C, N, H, W]
        select_features = (torch.sum(center_points_features * gaussian_maps, dim=1) / N ).unsqueeze(0)
        '''
        # assert not torch.isnan(select_features).any() and not torch.isinf(select_features).any()
        gaussian_maps = torch.sum(gaussian_maps, dim=1).unsqueeze(0)
        gaussian_maps = (gaussian_maps > 0).float()
        gaussian_maps = gaussian_maps.expand(1, C, H, W)        
        return gaussian_maps