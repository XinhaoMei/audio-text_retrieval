#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import os
import argparse
import torch
from trainer.trainer import train
from tools.config_loader import get_config


if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    torch.backends.cudnn.enabled = False

    parser = argparse.ArgumentParser(description='Settings.')
    parser.add_argument('-n', '--exp_name', default='exp_name', type=str,
                        help='Name of the experiment.')
    parser.add_argument('-d', '--dataset', default='Clotho', type=str,
                        help='Dataset used')
    parser.add_argument('-l', '--lr', default=0.0001, type=float,
                        help='Learning rate')
    parser.add_argument('-c', '--config', default='settings', type=str,
                        help='Name of the setting file.')
    parser.add_argument('-o', '--loss', default='weight', type=str,
                        help='Name of the loss function.')
    parser.add_argument('-f', '--freeze', default='False', type=str,
                        help='Freeze or not.')
    parser.add_argument('-e', '--batch', default=24, type=int,
                        help='Batch size.')
    parser.add_argument('-m', '--margin', default=0.2, type=float,
                        help='Margin value for loss')
    parser.add_argument('-s', '--seed', default=20, type=int,
                        help='Training seed')

    args = parser.parse_args()

    config = get_config(args.config)

    config.exp_name = args.exp_name
    config.dataset = args.dataset
    config.training.lr = args.lr
    config.training.loss = args.loss
    config.training.freeze = eval(args.freeze)
    config.data.batch_size = args.batch
    config.training.margin = args.margin
    config.training.seed = args.seed
    train(config)
