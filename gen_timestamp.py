import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_lidar_info(data):
    result = {}
    for item in data:
        pointcloud_path = item.get('pointcloud_path', '')
        lidar_timestamp = item.get('pointcloud_timestamp', '')

        label_number = pointcloud_path.split('/')[-1].split('.')[0]

        result[label_number] = lidar_timestamp

    return result

def save_results(result, output_file):
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    json_file_path = 'dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/vehicle-side/data_info.json'

    data = load_json(json_file_path)
    lidar_info = extract_lidar_info(data)

    output_file_path = 'lidar_timestamps.json' 
    save_results(lidar_info, output_file_path)