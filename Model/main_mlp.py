#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/3 01:28
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : main_mlp.py
# @Software: PyCharm

import os
from Net.mlp import MLP
from Config.config import parser
from Data.loader import data_loader_mlp
import sys

from Interface import interface

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".." + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    args = parser.parse_args()

    exp_dict = {

                'scalar': [ 'nbjets', 'mH1', 'mH2', 'mH3',
                'dPhiH1', 'dPhiH2', 'dPhiH3', 'dRH1', 'dRH2', 'dRH3', 'pTH1', 'pTH2',
                'pTH3',   'circH1', 'circH2', 'circH3',
                'mHHH', 'rmsmH', 'rmsOverMeanH', 'cosAngleH2', 'cosAngleH3', 'eta_mHHH'],

                'label': ['isSignal']}

    train_loader = data_loader_mlp(
        file_path='/lustre/collider/wanghaoyu/Ntuples/triHiggs_ML_v2/training/triHiggs_ML.root',
        exp_dict=exp_dict,
        shuffle=True,
        tree_name='HHHNtuple',
        num_workers=0,

        batch_size=args.batch_size
    )


    validation_loader = data_loader_mlp(
        file_path='/lustre/collider/wanghaoyu/Ntuples/triHiggs_ML_v2/training/triHiggs_ML.root',
        exp_dict=exp_dict,
        shuffle=True,
        tree_name='HHHNtuple',
        num_workers=0,

        batch_size=args.batch_size,
        drop_last=False
    )

    net_name = '0403_{}_mc_mlp_epoch_{}_lr_{}_batch_{}_optim_{}_classes_{}_l_gamma_{}_step_{}_v1'.format(
        args.index,
        args.n_epoch,
        args.learning_rate,
        args.batch_size,
        args.optim,
        args.n_classes,
        args.l_gamma,
        args.step,

    )

    ckp_dir = os.path.join('/lustre/collider/songsiyuan/TriHiggs/CheckPoint', net_name)

    interface(hyper_para={'net_used': 'mlp',
                          'n_classes': args.n_classes,
                          'batch_size': args.batch_size,
                          'n_epoch': args.n_epoch,
                          'l_r': args.learning_rate,
                          'optim': args.optim,
                          'k': args.k,
                          'global_vector_channel':len(exp_dict['scalar']),

                          'dropout': 0.5,
                          'scheduler':'cos'
                          },

              loader_train=train_loader,
              loader_valid=validation_loader,

              net=MLP,
              # data_loader_func=loader.data_loader_gnn,

              ckp_dir=ckp_dir,
              eval_para={'root_dir': ckp_dir,
                         'n_classes': args.n_classes,
                         'loader_test': validation_loader,
                         'ana_dir_name': 'Validation',
                         'threshold': 0,
                         'threshold_num': 101,

                         },

              # ann_eval_para={'ckp_dir': ckp_dir,
              #                'data_dir_format': {},
              #                'n_classes': args.n_classes,
              #                'pid_data_loader_func': None,
              #                'max_nodes': args.max_nodes,
              #                'ann_signal_label_list': [0, 1],
              #                'effi_points': [0.95, 0.96, 0.97, 0.98, 0.99][::-1],
              #                },

              TRAIN=True,
              ck_ann_info=True

              )

    pass

