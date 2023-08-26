import os
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import RoIAlign

def patchify(features, patch_size):
    """
    features: (N, C, H, W)  28 x 64
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert features.shape[2] % p == 0 and features.shape[3] % p == 0 

    h, w = features.shape[2] // p, features.shape[3] // p, 
    x = features.reshape(shape=(features.shape[0], features.shape[1], h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(features.shape[0], h * w, p**2 * features.shape[1]))
    return x, (h, w)

def unpatchify(x, patch_size, hw):
    """
    x: (N, L, patch_size**2 *C)
    features: (N, C, H, W)
    """
    p = patch_size
    c = int(x.shape[2] / (patch_size**2))
    h, w = hw[0], hw[1]
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    features = x.reshape(shape=(x.shape[0], c, h * p, w * p))
    return features

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


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

    def random_zero_out(self, tensor, p=0.5):
        mask = torch.rand(tensor.shape) > p
        tensor[mask] = 0.0
        return tensor
    
    def patch_mask(self, gaussian_maps, patch_size=2, mask_ratio=0.5):
        x, hw = patchify(gaussian_maps, patch_size=patch_size)
        _, mask, _ = random_masking(x, mask_ratio=mask_ratio)
        mask = unpatchify(mask.unsqueeze(-1).repeat(1, 1, patch_size*patch_size), patch_size, hw)
        return gaussian_maps * mask

    def forward(self, pred_box_infra, infra_features, level, sample_idx):
        downsample_rate = self.downsample_rate * math.pow(2, level)
        _, C, H, W = infra_features.shape
        device = infra_features.device
        if pred_box_infra is None:
            return torch.zeros_like(infra_features), torch.zeros_like(infra_features)
        N = min(pred_box_infra.shape[0], 20)
        if N == 0:
            return torch.zeros_like(infra_features), torch.zeros_like(infra_features)

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
            return select_features, select_features
        center_points_3d_bev = torch.zeros([center_points_3d.shape[0], 2], dtype=center_points_3d.dtype).to(device)
        center_points_3d_bev[:, 0] = (center_points_3d[:, 0] - self.lidar_range[0]) / (self.voxel_size[0] * downsample_rate)
        center_points_3d_bev[:, 1] = (center_points_3d[:, 1] - self.lidar_range[1]) / (self.voxel_size[1] * downsample_rate)

        Y, X = torch.meshgrid([torch.arange(H, device=device), torch.arange(W, device=device)], indexing="ij") 
        gaussian_maps_list = []
        for i in range(N):
            init_sigma = 1
            sigma = init_sigma * math.pow(2, -1 * level)
            gaussian_map = torch.exp((-(X - center_points_3d_bev[i][0])**2 - (Y - center_points_3d_bev[i][1])**2) / (sigma**2))
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
        if gaussian_maps.shape[2] == 100 and False:
            gaussian_maps_demo = gaussian_maps[0].detach().cpu().numpy()[0] * 100            
            cv2.imwrite(os.path.join("demo", "infra_features_demo_" + str(sample_idx.cpu().numpy()) + ".jpg"), gaussian_maps_demo)
        # gaussian_maps[:, :, :, :] = 1.0
        # gaussian_maps = self.random_zero_out(gaussian_maps, p=0.05)
        if gaussian_maps.shape[2] == 100 and False:
            gaussian_maps_demo = gaussian_maps[0].detach().cpu().numpy()[0] * 100
            cv2.imwrite(os.path.join("demo", "infra_features_demo_zero_out_" + str(sample_idx.cpu().numpy()) + ".jpg"), gaussian_maps_demo)
        # print("1. ", torch.sum(gaussian_maps) / (1 * H * W))
        gaussian_maps = gaussian_maps.expand(1, C, H, W)
        gaussian_maps_masked = self.patch_mask(gaussian_maps, patch_size=2, mask_ratio=0.5)

        # print("2. ", torch.sum(gaussian_maps) / (C * H * W))
        return gaussian_maps, gaussian_maps_masked

'''
if __name__ == "__main__":
    print("hello world...")
    img_path = "datasets/dair-v2x-c/training/vehicle-side/image/000002.jpg"
    img = cv2.imread(img_path)
    features = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    print(features.shape)

    patch_size = 40
    x, hw = patchify(features, patch_size=patch_size)
    x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.5)
    print(x_masked.shape, mask.unsqueeze(-1).shape, ids_restore.shape, mask.unsqueeze(-1).repeat(1, 1, patch_size*patch_size))
    features_masked = unpatchify(mask.unsqueeze(-1).repeat(1, 1, patch_size*patch_size), patch_size, hw)

    features = unpatchify(x, patch_size, hw)

    print("11", features_masked[0,0].shape, features.shape, features[0].permute(1,2,0).shape)
    cv2.imwrite("mask.jpg", features_masked[0,0].numpy() * 255)
    cv2.imwrite("features.jpg", features[0].permute(1,2,0).numpy())
'''