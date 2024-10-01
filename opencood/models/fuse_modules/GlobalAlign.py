import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class GlobalAlign(nn.Module):
    def __init__(self, in_channel=64) -> None:
        super(GlobalAlign, self).__init__()
        in_channel = in_channel*2
        self.offset_conv = nn.Conv2d(in_channel, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, infra_fea, vehicle_fea):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
           
        if  self.training:
            shift_x = random.randint(0, 5)
            shift_y = random.randint(0, 5)
        else:
            shift_x = 0
            shift_y = 0
            
        shifted_infra_fea = torch.roll(infra_fea, shifts=(shift_x, shift_y), dims=(3, 2)) # infra_fea  add noise
        offset = self.offset_conv(torch.cat([shifted_infra_fea, vehicle_fea], dim=1)) # concate in channel
        offset = offset.permute(0, 2, 3, 1)
        deform_weight = F.grid_sample(shifted_infra_fea, offset) 
        cor_infra_frea = shifted_infra_fea * deform_weight
       
        return cor_infra_frea
    
    def calculate_loss(self, deformed_feature, mm_bev):
        loss_fn = nn.MSELoss()
        loss = loss_fn(deformed_feature, mm_bev)
        return loss


