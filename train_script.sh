# CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet.yaml
# python opencood/tools/inference.py --model_dir opencood/logs/dair_where2comm_max_multiscale_resnet_2024_08_16_08_25_05  --fusion_method  intermediate_with_comm


# python opencood/tools/inference.py --model_dir opencood/logs/mae_logs/dair_where2comm_max_multiscale_resnet_2024_08_26_22_21_59 --fusion_method  intermediate_with_comm

# CUDA_VISIBLE_DEVICES=1 python opencood/tools/train_single.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet.yaml --model_dir opencood/logs/mae_logs/dair_where2comm_max_multiscale_resnet_2024_08_26_09_07_27


# no_mae
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_single.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_infra.yaml


# mae
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_single.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_vehicle.yaml