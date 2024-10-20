import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from opencood.models.fuse_modules.feature_flow import FlowGenerator

class How2commPreprocess(nn.Module):
    def __init__(self, args):
        super(How2commPreprocess, self).__init__()
        self.flow = FlowGenerator(args)

    def get_grid(self, flow):
        m, n = flow.shape[-2:]
        shifts_x = torch.arange(
            0, n, 1, dtype=torch.float32, device=flow.device)
        shifts_y = torch.arange(
            0, m, 1, dtype=torch.float32, device=flow.device)
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)

        grid_dst = torch.stack((shifts_x, shifts_y)).unsqueeze(0)
        workspace = torch.tensor(
            [(n - 1) / 2, (m - 1) / 2]).view(1, 2, 1, 1).to(flow.device)

        flow_grid = ((flow + grid_dst) / workspace - 1).permute(0, 2, 3, 1)

        return flow_grid

    def resample(self, feats, flow):
        flow_grid = self.get_grid(flow)
        warped_feats = F.grid_sample(
            feats, flow_grid, mode="bilinear", padding_mode="border")

        return warped_feats

    def forward(self, feat_curr, feat_history, feat_future, time):
        
        feat_history.reverse()
        
        B = len(feat_curr)
        feat_list = [[]] * B
        feat_history_list = [  torch.cat([feat_history[i][j] for i in range(len(feat_history))], dim=0)  for j in range(B)]
        
        for bs in range(B):
            feat_list[bs] = [feat_history_list[bs], feat_curr[bs], feat_future[bs]]

        feat_final, offset_loss = self.flow(feat_list, time)

        return feat_final, offset_loss
    