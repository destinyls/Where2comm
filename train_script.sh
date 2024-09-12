
# no_mae 路端
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_infra.yaml

# mae 车端
# CUDA_VISIBLE_DEVICES=3 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_vehicle1.yaml

CUDA_VISIBLE_DEVICES=2 python opencood/tools/inference_.py --model_dir opencood/logs/mae_logs/dair_where2comm_max_multiscale_resnet_2024_09_10_14_04_43  --fusion_method intermediate_with_comm