import os
import json

co_label_pth = "/home/yanglei/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/testing/cooperative/label_world"
veh_label_path = "/home/yanglei/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/testing/vehicle-side/label/lidar" 
inf_label_path = "/home/yanglei/Dataset/DAIR-V2X-C/cooperative-vehicle-infrastructure/testing/infrastructure-side/label/virtuallidar"
if __name__ == "__main__":
    os.makedirs(co_label_pth, exist_ok=True)
    os.makedirs(veh_label_path, exist_ok=True)
    os.makedirs(inf_label_path, exist_ok=True)

    for idx in range(30000):
        with open(os.path.join(co_label_pth, "{:06d}".format(idx)) + ".json", 'w') as fp:
            json.dump([], fp)
        with open(os.path.join(veh_label_path, "{:06d}".format(idx)) + ".json", 'w') as fp:
            json.dump([], fp)
        with open(os.path.join(inf_label_path, "{:06d}".format(idx)) + ".json", 'w') as fp:
            json.dump([], fp)
