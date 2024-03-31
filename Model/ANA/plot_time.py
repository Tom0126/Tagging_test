#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 19:43
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : plot_time.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


def plot_time(time_dict: dict, save_path: str, event_number: int, batch_list: list, **kwargs):
    epoch = lambda x: int((re.search(r"epoch_(\d+)_?", x)).group(1))
    batch = lambda x: int((re.search(r"batch_(\d+)_?", x)).group(1))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fontsize = 15

    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    num_bar = len(time_dict.keys())

    bar_width = kwargs.get('bar_width', 0.8 / num_bar / 2)

    time_max = []
    inference_max = []
    col_num = 0
    for key, value in time_dict.items():
        training = [((pd.read_csv(i))[kwargs.get('col', 'time')].values)[0] / (epoch(i)) for i in value]

        x_base = np.linspace(1, len(value), len(value))
        ax1.bar(x_base + (col_num - num_bar + .5) * bar_width,
                training,
                label=key + ' training',
                width=bar_width,
                color=colors[col_num]
                )

        time_max.append(np.amax(training))

        inference = [((pd.read_csv(i))[kwargs.get('col', 'time')].values)[0] * batch(i) / epoch(i) / event_number for i
                     in value]

        ax2.bar(x_base + (col_num + 0.5) * bar_width,
                inference,
                label=key + ' inference',
                width=bar_width,
                color=colors[col_num],
                alpha=0.4
                )

        col_num += 1

        inference_max.append(np.amax(inference))

        print(inference)

    ax1.set_ylabel('Training time [s/epoch]', fontsize=fontsize)
    ax1.set_ylim(top=1.38 * max(time_max))

    ax1.tick_params(axis='both', labelsize=13)

    ax2.set_ylabel('Inference time [s]', fontsize=fontsize)
    ax2.set_ylim(top=1.3 * max(inference_max))
    ax2.tick_params(axis='both', labelsize=13)

    plt.xticks(np.linspace(1, len(batch_list), len(batch_list)), list(map(lambda x: str(x), batch_list)), fontsize=12)
    ax1.set_xlabel('Batch size', fontsize=fontsize)

    ax1.legend(loc='upper left', bbox_to_anchor=(0.05, 0.99))
    ax2.legend(loc='upper left', bbox_to_anchor=(0.35, 0.99))

    plt.savefig(save_path)

    plt.show()
    plt.close(fig)


def plot_time_event(time_dict: dict, save_path: str, event_number: int, batch_list: list, **kwargs):
    epoch = lambda x: int((re.search(r"epoch_(\d+)_?", x)).group(1))
    batch = lambda x: int((re.search(r"batch_(\d+)_?", x)).group(1))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fontsize = 15

    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    num_bar = len(time_dict.keys())

    bar_width = kwargs.get('bar_width', 0.8 / num_bar)

    time_max = []
    inference_max = []
    col_num = 0
    for key, value in time_dict.items():
        training = [((pd.read_csv(i))[kwargs.get('col', 'time')].values)[0] / (epoch(i)) for i in value]

        x_base = np.linspace(1, len(value), len(value))
        ax1.bar(x_base + (col_num - num_bar/2+0.5) * bar_width,
                training,
                label=key,
                width=bar_width,
                color=colors[col_num]
                )

        print(training)

        time_max.append(np.amax(training))

        inference = [((pd.read_csv(i))[kwargs.get('col', 'time')].values)[0] / epoch(i) / event_number*1000 for i
                     in value]

        ax2.bar(x_base + (col_num - num_bar/2+0.5) * bar_width,
                inference,
                width=bar_width,
                color=colors[col_num],
                alpha=1
                )

        col_num += 1

        inference_max.append(np.amax(inference))

        # print(inference)

    ax1.set_ylabel('Training time [s/epoch]', fontsize=fontsize)
    ax1.set_ylim(top=1.1 * max(time_max))

    ax1.tick_params(axis='both', labelsize=13)

    ax2.set_ylabel('Inference time [ms/event]', fontsize=fontsize)
    ax2.set_ylim(top=1.1 * max(inference_max))
    ax2.tick_params(axis='both', labelsize=13)

    plt.xticks(np.linspace(1, len(batch_list), len(batch_list)), list(map(lambda x: str(x), batch_list)), fontsize=12)
    ax1.set_xlabel('Batch size', fontsize=fontsize)

    ax1.legend(loc='upper right', bbox_to_anchor=(0.97, 1), fontsize=14)


    plt.savefig(save_path)

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    time_dict = {
        'ResNet': [
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_33_mc_resnet_avg_epoch_200_lr_0.0001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_13_mc_resnet_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_17_mc_resnet_avg_epoch_200_lr_0.0001_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_39_mc_resnet_avg_epoch_50_lr_0.0001_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/time.csv'
        ],
        'DGRes': [
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_67_mc_dgres_epoch_30_lr_0.01_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_10_fk_3_fs_1_fp_1_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_78_mc_dgres_epoch_30_lr_0.001_batch_64_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_10_fk_3_fs_1_fp_1_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_79_mc_dgres_epoch_30_lr_0.001_batch_128_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_10_fk_3_fs_1_fp_1_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119_95_mc_dgcnn_epoch_30_lr_0.1_batch_256_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_128_k_20_v1/time.csv'
        ],

        'DGCNN': [
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119_31_mc_dgcnn_epoch_30_lr_0.1_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_128_k_20_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119_43_mc_dgcnn_epoch_30_lr_0.1_batch_64_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_128_k_20_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119_55_mc_dgcnn_epoch_30_lr_0.1_batch_128_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_128_k_20_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119_30_mc_dgcnn_epoch_30_lr_0.1_batch_256_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_64_k_20_v1/time.csv',
        ],
        'GravNet': [
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0122_29_mc_gravnet_epoch_30_lr_0.001_batch_32_optim_SGD_classes_2_nodes_256_k_10_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0122_39_mc_gravnet_epoch_30_lr_0.001_batch_64_optim_SGD_classes_2_nodes_256_k_10_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0122_40_mc_gravnet_epoch_30_lr_0.001_batch_128_optim_SGD_classes_2_nodes_256_k_10_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0122_41_mc_gravnet_epoch_30_lr_0.001_batch_256_optim_SGD_classes_2_nodes_256_k_10_v1/time.csv',

        ],
        'AlexNet': [
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0116_11_mc_alexnet_epoch_30_lr_0.001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0116_10_mc_alexnet_epoch_30_lr_0.001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0117_12_mc_googlenet_epoch_30_lr_0.001_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0117_11_mc_googlenet_epoch_30_lr_0.001_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_v1/time.csv',

        ],

        'LeNet': [
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0118_12_mc_lenet_epoch_200_lr_1e-05_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0118_13_mc_lenet_epoch_200_lr_1e-05_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0118_14_mc_lenet_epoch_200_lr_1e-05_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_v1/time.csv',
            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0118_39_mc_lenet_epoch_200_lr_1e-05_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_v1/time.csv',

        ],
    }

    event_number = 19200

    # plot_time(time_dict=time_dict,
    #           save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/time_cost.png',
    #           event_number=event_number,
    #           batch_list=[32, 64, 128, 256],
    #           col='train_time'
    #           # bar_width=0.1,
    #           )

    plot_time_event(time_dict=time_dict,
                    save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/time_cost_event.png',
                    event_number=event_number,
                    batch_list=[32, 64, 128, 256],
                    col='train_time'
                    # bar_width=0.1,
                    )

    pass
