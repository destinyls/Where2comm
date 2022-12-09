import os
import json
import pickle
import torch

import numpy as np

from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import tfm_to_pose, x1_to_x2, x_to_world

from opencood.data_utils.datasets.intermediate_fusion_dataset_dair import load_json
from opencood.visualization import simple_vis
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum

calib_path = "/root/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/testing/vehicle-side/calib/lidar_to_novatel"
npy_path = "/root/Where2comm/opencood/logs/dair_where2comm_attn_multiscale_resnet_2022_12_07_10_16_42/npy"
test_path = "/root/test"
test_json = "/root/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/testing/test.json"

co_annos_json = "/root/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/testing/cooperative/label_world/001482.json"
lidar_to_novatel = "/root/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/testing/vehicle-side/calib/lidar_to_novatel/001482.json"
novatel_to_world = "/root/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/testing/vehicle-side/calib/novatel_to_world/001482.json"
pcd_path = "/root/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/testing/vehicle-side/velodyne/001482.pcd"
npy_pred_path = "opencood/logs/dair_where2comm_attn_multiscale_resnet_2022_12_07_10_16_42/npy/1482_pred.npy"
pkl_pred_path = "/root/result/001482.pkl"
json_pred_path = "/root/001482.json"

def encode_corners3d(ry, dims, locs):
    l, h, w = dims[0], dims[1], dims[2]
    x, y, z = locs[0], locs[1], locs[2]
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

def json2corners_veh(json_path, RT=np.eye(4)):
    with open(json_path,'r',encoding='utf8') as fp:
        annos = json.load(fp)
    gt_boxes3d = []
    for idx in range(len(annos)):
        if annos[idx]["type"] not in ["car", "Car"]:
            continue
        dim = annos[idx]["3d_dimensions"]
        dim = [dim['l'], dim['w'], dim['h']]
        loc = annos[idx]["3d_location"]
        loc = [loc['x'], loc['y'], loc['z']]
        rot = annos[idx]["rotation"]
        box3d = encode_corners3d(rot, dim, loc)
        box3d = np.concatenate((box3d, np.ones((8, 1))), axis=1)
        box3d = np.matmul(RT, box3d.T).T[:, :3]
        gt_boxes3d.append(box3d[np.newaxis, :, :])
    gt_boxes3d = np.concatenate(gt_boxes3d, axis=0)  
    return gt_boxes3d

def json2corners_co(json_path, RT=np.eye(4)):
    with open(json_path,'r',encoding='utf8') as fp:
        annos = json.load(fp)
    gt_boxes3d = []
    for idx in range(len(annos)):
        world_8_points = np.array(annos[idx]["world_8_points"])
        world_8_points = np.concatenate((world_8_points, np.ones((8, 1))), axis=1)
        lidar_8_points = np.matmul(RT, world_8_points.T).T[:, :3]
        gt_boxes3d.append(lidar_8_points[np.newaxis, :, :])
    gt_boxes3d = np.concatenate(gt_boxes3d, axis=0)  
    return gt_boxes3d

def calib_json2RT(calib_json):
    with open(calib_json,'r',encoding='utf8') as fp:
        calib_velo2ego = json.load(fp)
    transform = calib_velo2ego["transform"] if "transform" in calib_velo2ego.keys() else calib_velo2ego
    translation = np.array(transform["translation"]).squeeze()
    rotation = np.array(transform["rotation"]).reshape(3, 3)
    RT = np.eye(4)
    RT[:3, :3] = rotation
    RT[:3, 3] = translation
    return RT

if __name__ == "__main__":
    transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(load_json(lidar_to_novatel), load_json(novatel_to_world))
    lidar_pose = tfm_to_pose(transformation_matrix)
    lidar2world = x_to_world(lidar_pose) # T_world_lidar
    world2lidar = np.linalg.inv(lidar2world)

    # gt_boxes3d_co = json2corners_co(co_annos_json, world2lidar)
    # print("2. ", gt_boxes3d_co)

    # gt_boxes3d_co_tensor = torch.tensor(gt_boxes3d_co).cuda()
    pred_box3d = np.load(npy_pred_path)
    pred_box3d_tensor = torch.tensor(pred_box3d).cuda()
    print(pred_box3d_tensor.shape)

    with open(pkl_pred_path, 'rb') as f:
        pred_box3d_baseline = pickle.load(f)
    print(pred_box3d_baseline.keys(), pred_box3d_baseline["veh_id"])
    inf_boxes = pred_box3d_baseline["boxes_3d"]
    inf_boxes_tensor = torch.tensor(inf_boxes).cuda()
    json_pred = load_json(json_pred_path)
    json_boxes = np.array(json_pred["boxes_3d"]).reshape(-1, 8, 3)
    json_boxes_tensor = torch.tensor(json_boxes).cuda()

    print(json_boxes_tensor.shape, pred_box3d_tensor.shape)

    gt_range = [0, -40, -3, 102.4, 40, 1]
    lidar_np, _ = pcd_utils.read_pcd(pcd_path)
    lidar_np_clean = mask_points_by_range(lidar_np, gt_range)
    lidar_np_clean_tensor = torch.tensor(lidar_np_clean).cuda()
    vis_save_path = "demo_3d.png"
    simple_vis.visualize(pred_box3d_tensor,
                        inf_boxes_tensor,
                        lidar_np_clean_tensor,
                        gt_range,
                        vis_save_path,
                        method='bev',
                        left_hand=False,
                        vis_pred_box=True)
                
    with open(test_json,'r',encoding='utf8') as fp:
        test_list = json.load(fp)
    print(len(test_list))
    for frame_id in test_list:
        '''
        if frame_id not in ["001482"]:
            continue
        '''
        calib_file = os.path.join(calib_path, frame_id + ".json")
        with open(calib_file,'r',encoding='utf8') as fp:
            calib = json.load(fp)
        transform = calib["transform"]
        translation = np.array(transform["translation"]).squeeze()
        rotation = np.array(transform["rotation"]).reshape(3,3)
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = rotation
        lidar2ego[:3, 3] = translation
        ego2lidar = np.linalg.inv(lidar2ego)

        result_dict = dict()
        if os.path.exists(os.path.join(npy_path, str(int(frame_id)) + "_pred.npy")):
            npy_pred_file = os.path.join(npy_path, str(int(frame_id)) + "_pred.npy")
            npy_pred_score_file = os.path.join(npy_path, str(int(frame_id)) + "_pred_score.npy")
            pred_box3d = np.load(npy_pred_file)
            pred_score = np.load(npy_pred_score_file)

            boxes_3d, scores_3d, labels_3d = [], [], []
            for i in range(pred_box3d.shape[0]):
                corner3d_ego = pred_box3d[i].astype(np.float16)
                score = pred_score[i]
                boxes_3d.append(corner3d_ego[np.newaxis, :, :])
                scores_3d.append(round(float(score), 2))
                labels_3d.append(int(2))
            if len(boxes_3d) > 0:
                boxes_3d = np.concatenate(boxes_3d, axis=0)
                boxes_3d = boxes_3d.tolist()
            result_dict["boxes_3d"] = boxes_3d
            result_dict["labels_3d"] = labels_3d
            result_dict["scores_3d"] = scores_3d
            result_dict["ab_cost"] = 100
        else:
            result_dict = {"boxes_3d": [], "labels_3d": [], "scores_3d": [], "ab_cost": 100}

        with open(os.path.join(test_path, frame_id + ".json"), 'w') as fp:
            json.dump(result_dict, fp)
