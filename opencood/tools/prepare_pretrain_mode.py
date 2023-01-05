import argparse

import os
import json
import numpy as np

from tqdm import tqdm

def parser():
    parser = argparse.ArgumentParser(description="pretrain data generation")
    parser.add_argument("--data_info_vehicle", default='/home/yanglei/DataSets/DAIR-V2X/cooperative-vehicle-infrastructure/training/vehicle-side/data_info.json', type=str,
                        help='path to data_info.json for vehicle-side')
    parser.add_argument("--data_info_infra", default='/home/yanglei/DataSets/DAIR-V2X/cooperative-vehicle-infrastructure/training/infrastructure-side/data_info.json', type=str,
                        help='path to data_info.json for infra-side')
    parser.add_argument("--data_info_cooperative", default='/home/yanglei/DataSets/DAIR-V2X/cooperative-vehicle-infrastructure/training/cooperative_pretrain/data_info.json', type=str,
                        help='path to save data_info.json for cooperative')
    parser.add_argument("--data_split", default='/home/yanglei/DataSets/DAIR-V2X/cooperative-vehicle-infrastructure/training/train_pretrain.json', type=str,
                        help='path to save data split train.json')
    opt = parser.parse_args()
    return opt

def encode_corners(ry, dims, locs):
    l, h, w = dims['l'], dims['h'], dims['w']
    x, y, z = locs['x'], locs['y'], locs['z']

    x_corners = [0, l, l, l, l, 0, 0, 0]
    y_corners = [0, 0, h, h, 0, 0, h, h]
    z_corners = [0, 0, 0, w, w, w, w, 0]

    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)
    z_corners += - np.float32(w) / 2

    corners_3d = np.array([x_corners, y_corners, z_corners])
    rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
    corners_3d = np.matmul(rot_mat, corners_3d)
    corners_3d += np.array([x, y, z]).reshape([3, 1])
    return corners_3d.T

def generate_data_info(opt):
    with open(opt.data_info_vehicle, "rb") as f:
        data_info_v = json.load(f)
    with open(opt.data_info_infra, "rb") as f:
        data_info_i = json.load(f)
    infra_pc_ids = list()
    for info in data_info_i:
        pc_id = int(info["pointcloud_path"].split('/')[1].split('.')[0])
        infra_pc_ids.append(pc_id)
    
    data_info_c, split_train = list(), list()
    infra_count = 0
    for info in data_info_v:
        pc_id_v = int(info["pointcloud_path"].split('/')[1].split('.')[0])
        pc_id_i = infra_pc_ids[infra_count]
        infra_count = infra_count + 1
        if infra_count == len(infra_pc_ids):
            infra_count = 0
            
        info_c = {"infrastructure_image_path": "infrastructure-side/image/" + "{:06d}".format(pc_id_i) + ".jpg",
                  "infrastructure_pointcloud_path": "infrastructure-side/velodyne/" + "{:06d}".format(pc_id_i) + ".pcd",
                  "vehicle_image_path": "vehicle-side/image/" + "{:06d}".format(pc_id_v) + ".jpg",
                  "vehicle_pointcloud_path": "vehicle-side/velodyne/"  + "{:06d}".format(pc_id_v) + ".pcd",
                  "cooperative_label_path": "cooperative_pretrain/label_world/" + "{:06d}".format(pc_id_v) + ".json",
                  "system_error_offset": {"delta_x": 0, "delta_y": 0}
                }   
        data_info_c.append(info_c)
        split_train.append("{:06d}".format(pc_id_v))
    with open(opt.data_info_cooperative, "w") as f:
        json.dump(data_info_c, f)
    with open(opt.data_split, "w") as f:
        json.dump(split_train, f)
            
def generate_label_world(opt):
    src_label_path_v = os.path.join(opt.data_info_vehicle.split('data_info.json')[0], "label", "lidar")
    print(src_label_path_v)
    dest_label_world_ath = os.path.join(opt.data_info_cooperative.split('data_info.json')[0], "label_world")
    print(dest_label_world_ath)
    
    for label_name in tqdm(os.listdir(src_label_path_v)):
        src_label_file = os.path.join(src_label_path_v, label_name)
        dest_label_file = os.path.join(dest_label_world_ath, label_name)
        with open(src_label_file, 'rb') as f:
            src_label_list = json.load(f)    
        dest_label_list = list()
        for src_label in src_label_list:
            dim, loc, roty = src_label["3d_dimensions"], src_label["3d_location"], src_label["rotation"]
            world_8_points = encode_corners(roty, dim, loc)
            world_8_points = world_8_points.tolist()
            src_label["world_8_points"] = world_8_points
            src_label["type"] = src_label["type"].lower()
            dest_label_list.append(src_label)
        with open(dest_label_file, 'w') as f:
            json.dump(dest_label_list, f)
        

if __name__ == "__main__":
    opt = parser()    
    generate_data_info(opt)
    # generate_label_world(opt)
    
    

        
