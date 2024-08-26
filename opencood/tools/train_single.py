# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yue Hu <18671129361@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
import datetime
import random
import warnings
import statistics

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import torch.backends.cudnn as cudnn

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Node rank for distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    opt = parser.parse_args()
    return opt

def main_worker(local_rank, nprocs, opt):

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    
    print('Creating Model')
    model = train_utils.create_model(hypes)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    cudnn.benchmark = True

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)
    
    bs = int(hypes['train_params']['batch_size'] / 1)
    distributed = False
    train_sampler, val_sampler = None, None
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,  device_ids=[local_rank], find_unused_parameters=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(opencood_train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(opencood_validate_dataset)
        
    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=bs,
                              num_workers=8,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              # shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=bs,
                            num_workers=8,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            # shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            sampler=val_sampler)        
    
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    '''

    # define the loss
    criterion = train_utils.create_loss(hypes)
    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model.module if distributed else model)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
    else:
        if hypes['resume'] is not None:
            model = train_utils.load_model(hypes['resume'], model.module if distributed else model)
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes, local_rank)
        # lr scheduler setup
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate
    with_round_loss = False
    mean_batch_time = 0.0
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        for i, batch_data in enumerate(train_loader):
            start_batch_time = time.time()
            if batch_data is None:
                continue
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_local_rank(batch_data, local_rank)
            # case1 : late fusion train --> only ego needed,
            # and ego is (random) selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            batch_data['ego']['epoch'] = epoch
            output_dict = model(batch_data['ego'], opencood_train_dataset)
            # first argument is always your output dictionary,
            # second argument is always your label dictionary.
            final_loss, single_loss_i, single_loss_v = 0.0, 0.0, 0.0
            if 'psm' in output_dict.keys():
                final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
            if len(output_dict) > 2:
                if 'loss_mae' in output_dict.keys():
                    loss_mae = output_dict['loss_mae']
                if 'psm_single_v' in output_dict.keys():
                    single_loss_v = criterion(output_dict, batch_data['ego']['label_dict_single_v'], prefix='_single_v')
                if 'psm_single_i' in output_dict.keys():
                    single_loss_i = criterion(output_dict, batch_data['ego']['label_dict_single_i'], prefix='_single_i')
                if 'fusion_args' in hypes['model']['args']:
                    if 'communication' in hypes['model']['args']['fusion_args']:
                        comm = hypes['model']['args']['fusion_args']['communication']
                        if ('round' in comm) and comm['round'] > 1:
                            round_loss_v = 0
                            with_round_loss = True
                            for round_id in range(1, comm['round']):
                                round_loss_v += criterion(output_dict, batch_data['ego']['label_dict'], prefix='_v{}'.format(round_id))
            
            if len(output_dict) > 2:
                final_loss += single_loss_v + single_loss_i 
                if loss_mae is not None:
                    final_loss += loss_mae
                    criterion.add_loss_dict("mae_loss_single", loss_mae)
                if with_round_loss:
                    final_loss += round_loss_v

            if local_rank == 0:
                batch_time = time.time() - start_batch_time
                mean_batch_time = ((len(train_loader) * epoch + i) * mean_batch_time + batch_time) / (len(train_loader) * epoch + i + 1)
                eta_seconds = mean_batch_time * (len(train_loader) * epoches - (len(train_loader) * epoch + i))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                criterion.logging(epoch, i, len(train_loader), eta_string, writer, nprocs)

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

                    batch_data = train_utils.to_local_rank(batch_data, local_rank)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'], opencood_train_dataset)

                    final_loss, single_loss_i, single_loss_v = 0.0, 0.0, 0.0
                    if 'psm' in output_dict.keys():
                        final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
                    if len(output_dict) > 2:
                        if 'psm_single_v' in output_dict.keys():
                            single_loss_v = criterion(output_dict, batch_data['ego']['label_dict_single_v'], prefix='_single_v')
                        if 'psm_single_i' in output_dict.keys():
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

        if epoch % hypes['train_params']['save_freq'] == 0 and local_rank == 0 and epoch > 25:  #  因为epoch数太大   save_freq=5  仅保存50 epoch之后的
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)
    torch.cuda.empty_cache()
    run_test = True
    if run_test:
        # fusion_method = opt.fusion_method
        cmd = f"python /home/yanglei/code/Where2comm/opencood/tools/inference.py --model_dir {saved_path} --fusion_method intermediate_with_comm"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    opt = train_parser()

    if opt.seed is not None:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    nprocs = 1
    print("opt.local_rank: ", opt.local_rank)
    main_worker(opt.local_rank, nprocs, opt)
