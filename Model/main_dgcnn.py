#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/26 21:27
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : main_dgcnn.py
# @Software: PyCharm

import os
from Net.dgcnn import DGCNN_vector_cls, get_graph_feature
from Config.config import parser
from Data.loader import data_loader_gnn
import sys

from Interface import interface

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".." + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    args = parser.parse_args()


    vector_v1= ['jets_pt', 'jets_pt']

    vector_v2 = ['jets_pt', 'jets_eta', 'jets_phi', 'jets_E']

    var_v1=['nbjets', 'mH1', 'mH2', 'mH3',
                           'dPhiH1', 'dPhiH2', 'dPhiH3', 'dRH1', 'dRH2', 'dRH3', 'pTH1', 'pTH2',
                           'pTH3', 'circH1', 'circH2', 'circH3',
                           'mHHH', 'rmsmH', 'rmsOverMeanH', 'cosAngleH2', 'cosAngleH3', 'eta_mHHH']

    var_v2=['njets', 'nbjets', 'TotalJetsEnergy', 'TotalHiggsJetPt', 'mH1', 'mH2', 'mH3', 'phiH1', 'phiH2', 'phiH3',
                'dEtaH1', 'dEtaH2', 'dEtaH3', 'dPhiH1', 'dPhiH2', 'dPhiH3', 'dRH1', 'dRH2', 'dRH3', 'pTH1', 'pTH2',
                'pTH3', 'mHangle1', 'mHangle2', 'mHangle3', 'asymH1', 'asymH2', 'asymH3', 'circH1', 'circH2', 'circH3',
                'mHHH', 'rmsmH', 'rmsOverMeanH', 'cosAngleH1', 'cosAngleH2', 'cosAngleH3', 'eta_mHHH', ]


    exp_dict = {'vector': vector_v2,

                'scalar': var_v2,

                'label': ['isSignal']}

    train_loader = data_loader_gnn(
        file_path='/lustre/collider/wanghaoyu/Ntuples/triHiggs_ML_v4/training/triHiggs_ML.root',
        exp_dict=exp_dict,
        shuffle=True,
        tree_name='HHHNtuple',
        num_workers=0,
        max_nodes=args.max_nodes,
        padding=bool(args.padding),
        batch_size=args.batch_size
    )

    validation_loader = data_loader_gnn(
        file_path='/lustre/collider/wanghaoyu/Ntuples/triHiggs_ML_v4/validation/triHiggs_ML.root',
        exp_dict=exp_dict,
        shuffle=True,
        tree_name='HHHNtuple',
        num_workers=0,
        max_nodes=args.max_nodes,
        padding=bool(args.padding),
        batch_size=args.batch_size,
        drop_last=False
    )

    net_name = '0501_{}_mc_dgcnn_epoch_{}_lr_{}_batch_{}_optim_{}_classes_{}_l_gamma_{}_step_{}_nodes_{}_k_{}_v1'.format(
        args.index,
        args.n_epoch,
        args.learning_rate,
        args.batch_size,
        args.optim,
        args.n_classes,
        args.l_gamma,
        args.step,
        args.max_nodes,
        args.k
    )

    ckp_dir = os.path.join('/lustre/collider/songsiyuan/TriHiggs/CheckPoint', net_name)

    interface(hyper_para={'net_used': 'dgcnn',
                          'n_classes': args.n_classes,
                          'batch_size': args.batch_size,
                          'n_epoch': args.n_epoch,
                          'l_r': args.learning_rate,
                          'optim': args.optim,
                          'max_nodes': args.max_nodes,
                          # 'padding':args.padding,
                          'k': args.k,
                          'global_vector_channel': len(exp_dict['scalar']),
                          'in_channel': len(exp_dict['vector']),
                          'channels': {'paper': [64, 64, 128, 256]},
                          'kernels': {'paper': [1, 1, 1, 1]},
                          'bns': {'paper': [True, True, True, True]},
                          'acti': {'paper': [True, True, True, True]},
                          'get_f_func': {'paper': get_graph_feature},
                          'adaptive_pool': None,
                          'pool_out_size': None,
                          'emb_dims': 1024,
                          'dropout': 0.5,
                          'scheduler': 'cos'
                          },

              loader_train=train_loader,
              loader_valid=validation_loader,

              net=DGCNN_vector_cls,
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
