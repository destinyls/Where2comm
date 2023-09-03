# --nproc_per_node=1 on machine with one gpu
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet.yaml
CUDA_VISIBLE_DEVICES=2 python opencood/tools/inference.py --model_dir opencood/logs/dair_where2comm_max_multiscale_resnet/ --fusion_method intermediate_with_comm
