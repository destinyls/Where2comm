
# no_mae 路端
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_infra.yaml

# mae 车端
# CUDA_VISIBLE_DEVICES=2 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_vehicle.yaml 

CUDA_VISIBLE_DEVICES=4 python opencood/tools/inference_.py --model_dir opencood/logs/exp/flowPre_dair_where2comm_max_multiscale_resnet_2024_10_11_03_50_57 --fusion_method intermediate_with_comm

CUDA_VISIBLE_DEVICES=2 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_vehicle2.yaml 

