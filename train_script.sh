
# no_mae 路端
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_single.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_infra.yaml --model_dir opencood/logs/no_mae_logs/dair_where2comm_max_multiscale_resnet_2024_09_04_23_11_03


# mae 车端
# CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_vehicle1.yaml
/home/yanglei/code/Where2comm/opencood/logs/mae_logs/dair_where2comm_max_multiscale_resnet_2024_09_05_23_01_05

CUDA_VISIBLE_DEVICES=2 python opencood/tools/inference.py --model_dir opencood/logs/no_mae_logs/dair_where2comm_max_multiscale_resnet_2024_09_04_23_11_03 --fusion_method intermediate_with_comm