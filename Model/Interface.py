#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/26 20:08
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : Interface.py
# @Software: PyCharm



import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from Net.dgcnn import DGCNN_vector_cls, get_graph_feature
from Config.config import parser
from Data import loader
import sys
import pandas as pd
import time
import glob

from Train import train
from Evaluate import evaluate

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".." + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

net_path_dict = {
    'lenet': './Net/lenet.py',
    'gravnet': './Net/grav_net.py',
    'resnet': './Net/resnet.py',
    'dgcnn': './Net/dgcnn.py'
}


def get_net_para(net_used, **kwargs):
    net_para = None
    if net_used in ['lenet', 'googlenet', 'alexnet']:
        net_para = {'n_classes': kwargs.get('n_classes')}


    elif net_used == 'gravnet':
        # not use
        net_para = {'n_classes': kwargs.get('n_classes'),
                    'input_node': 4 * kwargs.get('max_nodes'),
                    'grav_out_channels': kwargs.get('grav_out_channels'),
                    'gravnet_module': list(kwargs.get('gravnet_module').values())[0]
                    if isinstance(kwargs.get('gravnet_module'), dict) else kwargs.get('gravnet_module'),
                    'global_exchange_block': list(kwargs.get('global_exchange_block').values())[0]
                    if isinstance(kwargs.get('global_exchange_block'), dict) else kwargs.get('global_exchange_block'),
                    }

    elif net_used == 'gravnet_v2':

        net_para = {'n_classes': kwargs.get('n_classes'),
                    'max_nodes': kwargs.get('max_nodes'),
                    'in_channel': kwargs.get('in_channel'),
                    'in_channel_factor': kwargs.get('in_channel_factor'),
                    'grav_inner_channel': kwargs.get('grav_inner_channel'),
                    'k': kwargs.get('k'),
                    'space_dimensions': kwargs.get('space_dimensions'),
                    'propagate_dimensions': kwargs.get('propagate_dimensions'),
                    'gravnet_module': list(kwargs.get('gravnet_module').values())[0]
                    if isinstance(kwargs.get('gravnet_module'), dict) else kwargs.get('gravnet_module'),
                    'global_exchange_block': list(kwargs.get('global_exchange_block').values())[0]
                    if isinstance(kwargs.get('global_exchange_block'), dict) else kwargs.get('global_exchange_block'),
                    }

    elif net_used == 'resnet':
        net_para = {'block': list(kwargs.get('block').values())[0] if isinstance(kwargs.get('block'), dict)
        else kwargs.get('block'),
                    'layers': list(kwargs.get('layers').values())[0] if isinstance(kwargs.get('layers'), dict)
                    else kwargs.get('layers'),
                    'num_classes': kwargs.get('n_classes'),
                    'start_planes': kwargs.get('start_planes'),
                    'first_kernal': kwargs.get('first_kernal'),
                    'first_stride': kwargs.get('first_stride'),
                    'first_padding': kwargs.get('first_padding'),
                    'short_cut': kwargs.get('short_cut'),
                    }

    elif net_used == 'dgcnn':
        net_para = {'k': kwargs.get('k'),
                    'in_channel': kwargs.get('in_channel'),
                    'global_vector_channel': kwargs.get('global_vector_channel'),
                    'channels': list(kwargs.get('channels').values())[0] if isinstance(kwargs.get('channels'), dict)
                    else kwargs.get('channels'),
                    'kernels': list(kwargs.get('kernels').values())[0] if isinstance(kwargs.get('kernels'), dict)
                    else kwargs.get('kernels'),
                    'bns': list(kwargs.get('bns').values())[0] if isinstance(kwargs.get('bns'), dict)
                    else kwargs.get('bns'),
                    'acti': list(kwargs.get('acti').values())[0] if isinstance(kwargs.get('acti'), dict)
                    else kwargs.get('acti'),
                    'get_f_func': list(kwargs.get('get_f_func').values())[0] if isinstance(kwargs.get('get_f_func'),
                                                                                           dict)
                    else kwargs.get('get_f_func'),
                    'adaptive_pool': kwargs.get('adaptive_pool'),
                    'pool_out_size': list(kwargs.get('pool_out_size').values())[0] if isinstance(
                        kwargs.get('pool_out_size'), dict)
                    else kwargs.get('pool_out_size'),
                    'emb_dims': kwargs.get('emb_dims'),
                    'dropout': kwargs.get('dropout'),
                    'num_classes': kwargs.get('n_classes'),
                    }

    elif net_used == 'dgres':

        net_para = {
            'PCNN_block': list(kwargs.get('PCNN_block').values())[0] if isinstance(kwargs.get('PCNN_block'), dict)
            else kwargs.get('PCNN_block'),
            'k': kwargs.get('k'),
            'in_channel': kwargs.get('in_channel'),
            'channels': list(kwargs.get('channels').values())[0] if isinstance(kwargs.get('channels'), dict)
            else kwargs.get('channels'),
            'kernels': list(kwargs.get('kernels').values())[0] if isinstance(kwargs.get('kernels'), dict)
            else kwargs.get('kernels'),
            'bns': list(kwargs.get('bns').values())[0] if isinstance(kwargs.get('bns'), dict)
            else kwargs.get('bns'),
            'acti': list(kwargs.get('acti').values())[0] if isinstance(kwargs.get('acti'), dict)
            else kwargs.get('acti'),
            'get_f_func': list(kwargs.get('get_f_func').values())[0] if isinstance(kwargs.get('get_f_func'),
                                                                                   dict)
            else kwargs.get('get_f_func'),
            'adaptive_pool': kwargs.get('adaptive_pool'),
            'pool_out_size': list(kwargs.get('pool_out_size').values())[0] if isinstance(
                kwargs.get('pool_out_size'), dict)
            else kwargs.get('pool_out_size'),
            'Res_block': list(kwargs.get('Res_block').values())[0] if isinstance(kwargs.get('Res_block'),
                                                                                 dict)
            else kwargs.get('Res_block'),

            'block': list(kwargs.get('block').values())[0] if isinstance(kwargs.get('block'), dict)
            else kwargs.get('block'),
            'layers': list(kwargs.get('layers').values())[0] if isinstance(kwargs.get('layers'), dict)
            else kwargs.get('layers'),
            'start_planes': kwargs.get('start_planes'),
            'first_kernal': kwargs.get('first_kernal'),
            'first_stride': kwargs.get('first_stride'),
            'first_padding': kwargs.get('first_padding'),
            'short_cut': kwargs.get('short_cut'),
            'num_classes': kwargs.get('n_classes'),

            }

    return net_para


def save_para(save_path: str,
              para: dict):
    filename = open(save_path, 'w')  # dict to txt
    for k, v in para.items():

        if isinstance(v, dict):
            filename.write(k + ':' + str(list(v.keys())[0]))
        else:
            filename.write(k + ':' + str(v))
        filename.write('\n')
    filename.close()


def interface(hyper_para: dict,
              net: torch.nn.Module,
              ckp_dir: str,
              loader_train,
              loader_valid,
              eval_para: dict,
              # ann_eval_para: dict,
              **kwargs,
              ):


    if kwargs.get('ck_ann_info', False) and os.path.exists(os.path.join(ckp_dir, 'ANA/ann_info_s_1_b_0.csv')):

        sys.exit(0)



    os.makedirs(ckp_dir, exist_ok=True)

    save_para(save_path=os.path.join(ckp_dir, 'hyper_para.txt'),
              para=hyper_para)


    net_used = hyper_para.get('net_used')
    net_para = None

    if net_used in ['lenet', 'googlenet', 'alexnet']:
        net_para = get_net_para(net_used=net_used,
                                n_classes=hyper_para.get('n_classes'))

    elif net_used == 'gravnet':
        # not use
        net_para = get_net_para(net_used=net_used,
                                n_classes=hyper_para.get('n_classes'),
                                max_nodes=hyper_para.get('max_nodes'),
                                grav_out_channels=hyper_para.get('grav_out_channels'),
                                gravnet_module=hyper_para.get('gravnet_module'),
                                global_exchange_block=hyper_para.get('global_exchange_block'),
                                )
    elif net_used == 'gravnet_v2':

        net_para = get_net_para(net_used=net_used,
                                n_classes=hyper_para.get('n_classes'),
                                max_nodes=hyper_para.get('max_nodes'),
                                in_channel=hyper_para.get('in_channel'),
                                in_channel_factor=hyper_para.get('in_channel_factor'),
                                grav_inner_channel=hyper_para.get('grav_inner_channel'),
                                k=hyper_para.get('k'),
                                space_dimensions=hyper_para.get('space_dimensions'),
                                propagate_dimensions=hyper_para.get('propagate_dimensions'),
                                gravnet_module=hyper_para.get('gravnet_module'),
                                global_exchange_block=hyper_para.get('global_exchange_block'),
                                )

    elif net_used == 'resnet':
        net_para = get_net_para(net_used=net_used,
                                n_classes=hyper_para.get('n_classes'),
                                block=hyper_para.get('block'),
                                layers=hyper_para.get('layers'),
                                start_planes=hyper_para.get('start_planes'),
                                first_kernal=hyper_para.get('first_kernal'),
                                first_stride=hyper_para.get('first_stride'),
                                first_padding=hyper_para.get('first_padding'),
                                short_cut=bool(hyper_para.get('short_cut'))
                                )
    elif net_used == 'dgcnn':
        net_para = get_net_para(net_used=net_used,
                                n_classes=hyper_para.get('n_classes'),
                                k=hyper_para.get('k'),
                                in_channel=hyper_para.get('in_channel'),
                                global_vector_channel = hyper_para.get('global_vector_channel'),
                                channels=hyper_para.get('channels'),
                                kernels=hyper_para.get('kernels'),
                                bns=hyper_para.get('bns'),
                                acti=hyper_para.get('acti'),
                                get_f_func=hyper_para.get('get_f_func'),
                                adaptive_pool=hyper_para.get('adaptive_pool'),
                                pool_out_size=hyper_para.get('pool_out_size'),
                                emb_dims=hyper_para.get('emb_dims'),
                                dropout=hyper_para.get('dropout'),
                                )

    elif net_used == 'dgres':
        net_para = get_net_para(net_used=net_used,
                                n_classes=hyper_para.get('n_classes'),
                                PCNN_block=hyper_para.get('PCNN_block'),
                                k=hyper_para.get('k'),
                                in_channel=hyper_para.get('in_channel'),
                                channels=hyper_para.get('channels'),
                                kernels=hyper_para.get('kernels'),
                                bns=hyper_para.get('bns'),
                                acti=hyper_para.get('acti'),
                                get_f_func=hyper_para.get('get_f_func'),
                                adaptive_pool=hyper_para.get('adaptive_pool'),
                                pool_out_size=hyper_para.get('pool_out_size'),
                                Res_block=hyper_para.get('Res_block'),
                                block=hyper_para.get('block'),
                                layers=hyper_para.get('layers'),
                                start_planes=hyper_para.get('start_planes'),
                                first_kernal=hyper_para.get('first_kernal'),
                                first_stride=hyper_para.get('first_stride'),
                                first_padding=hyper_para.get('first_padding'),
                                short_cut=bool(hyper_para.get('short_cut')),
                                )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = net(**net_para)

    if kwargs.get('TRAIN', True):
        optimizer_dict = {
            'SGD': optim.SGD(net.parameters(),
                             lr=hyper_para.get('l_r'),
                             momentum=0.9),
            'Adam': optim.AdamW(net.parameters(),
                                lr=hyper_para.get('l_r'),
                                betas=(0.9, 0.999),
                                weight_decay=4e-5)
        }

        optimizer = optimizer_dict.get(hyper_para.get('optim'))

        scheduler = None
        if hyper_para.get('scheduler', 'step') == 'step':

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                        step_size=hyper_para.get('step'),
                                                        gamma=hyper_para.get('l_gamma'))
        elif hyper_para.get('scheduler', 'step') == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                   T_max=hyper_para.get('n_epoch'),
                                                                   eta_min=1e-3)


        train(net=net,
              max_epoch=hyper_para.get('n_epoch'),
              optimizer=optimizer,
              scheduler=scheduler,
              device=device,
              loader_train=loader_train,
              loader_valid=loader_valid,
              ckp_dir=ckp_dir,
              log_interval=hyper_para.get('log_interval', 1000),
              val_interval=hyper_para.get('val_interval', 5),
              )

    if kwargs.get('EVAL', True):
        evaluate(net=net,
                 **eval_para
                 )
    #
    # if kwargs.get('ANN_EVAL', True):
    #     ann_eval(net=net,
    #              **ann_eval_para)


if __name__ == '__main__':


    pass
