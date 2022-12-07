import json
import os

train_json = "/home/yanglei/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/train.json"
val_json = "/home/yanglei/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/val.json"
trainval_json = "/home/yanglei/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/trainval.json"
test_json = "/home/yanglei/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/testing/test.json"

test_path = "/home/yanglei/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/testing/vehicle-side/velodyne"


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    with open(train_json,'r',encoding='utf8') as fp:
        train_list = json.load(fp)
    with open(val_json,'r',encoding='utf8') as fp:
        val_list = json.load(fp)

    trainval_list = train_list + val_list
    with open(trainval_json, 'w') as fp:
        json.dump(trainval_list, fp)

    test_list = []
    co_datainfo = load_json("/home/yanglei/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/testing/cooperative/data_info.json")
    for frame_info in co_datainfo:
        veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
        test_list.append(veh_frame_id)
    with open(test_json, 'w') as fp:
        json.dump(test_list, fp)
    print("len: ", len(test_list))
    print("hello world ...")