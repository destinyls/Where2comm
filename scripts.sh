export PYTHONPATH=$PWD:$PYTHONPATH

# no_mae 路端  即 base
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_infra.yaml

# mae 车端
# CUDA_VISIBLE_DEVICES=2 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_vehicle.yaml 

# mask Train实验
CUDA_VISIBLE_DEVICES=8 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_for_mask.yaml 

# flow Train实验
CUDA_VISIBLE_DEVICES=9 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_for_flow.yaml 

# mask Inference实验
CUDA_VISIBLE_DEVICES=8 python opencood/tools/inference.py --model_dir opencood/logs/exp/onlyMask_dair_where2comm_max_multiscale_resnet_2024_12_12_09_47_17 --fusion_method intermediate_with_comm

# flow Inference实验
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_diff_delay.py --model_dir opencood/logs/exp/baseline_dair_where2comm_max_multiscale_resnet_2024_09_04_23_11_03 --fusion_method intermediate_with_comm