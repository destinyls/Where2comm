# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yue Hu <18671129361@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
import opencood.utils.misc as misc

from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from icecream import ic

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    
    # distributed training parameters
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    
    opt = parser.parse_args()
    return opt

def main(opt):
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    distributed = False  
    print("distributed: ", distributed)
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        num_tasks = torch.cuda.device_count()
        global_rank = torch.distributed.get_rank()    
        torch.cuda.set_device(global_rank)
        device = torch.device("cuda", global_rank)
    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(opencood_train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        validation_sampler = torch.utils.data.DistributedSampler(opencood_validate_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        train_loader = DataLoader(opencood_train_dataset,                              
                              sampler = train_sampler,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=16,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              pin_memory=True,
                              drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=validation_sampler,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=16,
                                collate_fn=opencood_train_dataset.collate_batch_train,                            
                                pin_memory=True,
                                drop_last=True)
    else:
        train_loader = DataLoader(opencood_train_dataset,                              
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=16,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              pin_memory=True,
                              drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=16,
                                collate_fn=opencood_train_dataset.collate_batch_train,                            
                                pin_memory=True,
                                drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)        
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[global_rank], output_device=global_rank, find_unused_parameters=True)
        model_without_ddp = model.module
                                                          
    # define the loss
    criterion = train_utils.create_loss(hypes)
    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model_without_ddp)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        # if misc.is_main_process():
        saved_path = train_utils.setup_train(hypes)
        # lr scheduler setup
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)
    
    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate
    with_round_loss = False
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        for i, batch_data in enumerate(train_loader):
            if batch_data is None:
                continue
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            # case1 : late fusion train --> only ego needed,
            # and ego is (random) selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            batch_data['ego']['epoch'] = epoch
            output_dict = model(batch_data['ego'])
            # first argument is always your output dictionary,
            # second argument is always your label dictionary.
            final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
            if len(output_dict) > 2:
                single_loss_v = criterion(output_dict, batch_data['ego']['label_dict_single_v'], prefix='_single_v')
                single_loss_i = criterion(output_dict, batch_data['ego']['label_dict_single_i'], prefix='_single_i')
                if 'fusion_args' in hypes['model']['args']:
                    if 'communication' in hypes['model']['args']['fusion_args']:
                        comm = hypes['model']['args']['fusion_args']['communication']
                        if ('round' in comm) and comm['round'] > 1:
                            round_loss_v = 0
                            with_round_loss = True
                            for round_id in range(1, comm['round']):
                                round_loss_v += criterion(output_dict, batch_data['ego']['label_dict'], prefix='_v{}'.format(round_id))

            if misc.is_main_process():
                criterion.logging(epoch, i, len(train_loader), writer)

            if len(output_dict) > 2:
                final_loss += single_loss_v + single_loss_i
                if with_round_loss:
                    final_loss += round_loss_v

            # back-propagation
            final_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    if len(output_dict) > 2:
                        single_loss_v = criterion(output_dict, batch_data['ego']['label_dict_single_v'], prefix='_single_v')
                        single_loss_i = criterion(output_dict, batch_data['ego']['label_dict_single_i'], prefix='_single_i')
                        final_loss += single_loss_v + single_loss_i

                        if 'fusion_args' in hypes['model']['args']:
                            if 'communication' in hypes['model']['args']['fusion_args']:
                                comm = hypes['model']['args']['fusion_args']['communication']
                                if ('round' in comm) and comm['round'] > 1:
                                    for round_id in range(1, comm['round']):
                                        round_loss_v = criterion(output_dict, batch_data['ego']['label_dict'], prefix='_v{}'.format(round_id))
                                        final_loss += round_loss_v
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:
            if misc.is_main_process():
                torch.save(model_without_ddp.state_dict(),
                        os.path.join(saved_path,
                                        'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)
    torch.cuda.empty_cache()
    run_test = True
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python /GPFS/data/yhu/code/OpenCOOD/opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    opt = train_parser()
    main(opt)
