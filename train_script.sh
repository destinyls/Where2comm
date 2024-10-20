
# no_mae 路端
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_infra.yaml

# mae 车端
# CUDA_VISIBLE_DEVICES=2 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_vehicle.yaml 

CUDA_VISIBLE_DEVICES=3 python opencood/tools/inference.py --model_dir opencood/logs/exp/branch_new_develflowPre_dair_where2comm_max_multiscale_resnet_2024_10_20_15_42_35 --fusion_method intermediate_with_comm

CUDA_VISIBLE_DEVICES=2 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_for_flow.yaml 

CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_singleGPU.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet_for_flow_finetune_head.yaml 

--model_dir opencood/logs/exp/flowPre_dair_where2comm_max_multiscale_resnet_2024_10_14_18_59_48
