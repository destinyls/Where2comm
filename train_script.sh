# --nproc_per_node=1 on machine with one gpu
python -m torch.distributed.launch --nproc_per_node=1 opencood/tools/train.py \
                    --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_where2comm_max_multiscale_resnet.yaml \
                    --resume /root/zhaiyize/yize/Where2comm/opencood/where2comm_infra/net_epoch69.pth
# python opencood/tools/inference.py --model_dir opencood/logs/dair_where2comm_attn_multiscale_resnet/ --fusion_method intermediate_with_comm