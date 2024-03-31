#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/26 22:18
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : plot_loss.py
# @Software: PyCharm
import glob
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(file_path_list,
              label_list,
              epoch_list,
              inter=1,
              **kwargs
              ):
    plt.figure(figsize=(6, 5))

    for file_path, label, epoch in zip(file_path_list, label_list, epoch_list):
        loss = pd.read_csv(file_path)

        print(len(loss['valid_x']))
        plt.plot(((loss['valid_x'].values) * epoch / (loss['valid_x'].values[-1]))[::inter],
                 (loss['valid_y'].values)[::inter],
                 label=label,
                 linewidth=2)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if kwargs.get('y_ul') != None:
        plt.ylim(0, kwargs.get('y_ul'))

    plt.xlim(right=max(epoch_list))
    plt.show()


def plot_loss_train(file_path_list,
                    label_list,
                    inter=1,
                    **kwargs
                    ):
    plt.figure(figsize=(6, 5))

    for file_path, label in zip(file_path_list, label_list):
        loss = pd.read_csv(file_path)

        print(len(loss['train_x']))
        plt.plot((loss['train_x'].values)[::inter],
                 (loss['train_y'].values)[::inter],
                 label=label,
                 linewidth=2)
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.xscale('log')
    if kwargs.get('y_ul') != None:
        plt.ylim(top=kwargs.get('y_ul'))

    if kwargs.get('x_ul') != None:
        plt.xlim(right=kwargs.get('x_ul'))

    plt.show()


def plot_error(file_path_list,
               label_list,
               save_path,
               inter=1,
               inter_val=1,
               x_scale=1,
               **kwargs
               ):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    plt.figure(figsize=(6, 5))

    for i, file_path, label in zip(range(len(file_path_list)), file_path_list, label_list):
        loss = pd.read_csv(file_path.get('train'))

        x = [j / x_scale for j in (loss['train_x'].values)[::inter]]
        x.append(loss['train_x'].values[-1] / x_scale)

        y = [j * 100 for j in (loss['train_error'].values)[::inter]]
        y.append((loss['train_error'].values)[-1] * 100)

        plt.plot(x,
                 y,
                 color=colors[i],
                 linewidth=1)

        loss = pd.read_csv(file_path.get('valid'))

        x = [j / x_scale for j in (loss['valid_x'].values)[::inter_val]]
        x.append(loss['valid_x'].values[-1] / x_scale)

        y = [j * 100 for j in (loss['valid_error'].values)[::inter_val]]
        y.append((loss['valid_error'].values)[-1] * 100)

        print(round((loss['valid_error'].values)[-1] * 100, 3))

        plt.plot(x,
                 y,
                 linestyle='-',
                 marker='*',
                 label=label,
                 color=colors[i],
                 linewidth=3,
                 markersize=16)

    for err in range(0, 50, 10):
        plt.plot(np.linspace(kwargs.get('x_ll', 0), kwargs.get('x_ul', 1000000), 100) / x_scale,
                 err * np.ones(100),
                 '--',

                 color='grey')

    if kwargs.get('y_log', False):
        plt.plot(np.linspace(kwargs.get('x_ll', 0), kwargs.get('x_ul', 1000000), 100) / x_scale,
                 1 * np.ones(100),
                 '--',

                 color='grey')

    plt.legend(fontsize=kwargs.get('fontsize', 13), loc='upper right')
    plt.xlabel('Iteration' if x_scale == 1 else r'$\mathrm{Iteration}\ (\times$' + '{})'.format(x_scale),
               fontsize=kwargs.get('fontsize', 14))
    plt.ylabel('Error [%]', fontsize=kwargs.get('fontsize', 14))

    plt.tick_params(labelsize=kwargs.get('fontsize', 13))

    if kwargs.get('y_log', False):
        plt.yscale('log')

    if kwargs.get('x_log', False):
        plt.xscale('log')

    if kwargs.get('y_ll') != None:
        plt.ylim(bottom=kwargs.get('y_ll'))

    if kwargs.get('y_ul') != None:
        plt.ylim(top=kwargs.get('y_ul'))

    if kwargs.get('x_ll') != None:
        plt.xlim(left=kwargs.get('x_ll') / x_scale)

    if kwargs.get('x_ul') != None:
        plt.xlim(right=kwargs.get('x_ul') / x_scale)
    plt.savefig(save_path)
    plt.show()
    print(label_list[1], '\n')


if __name__ == '__main__':
    inter_val = 25
    y_ul = 50
    y_log = True
    # 1e-5
    # file_path_list = [
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_161_mc_resnet_avg_epoch_200_lr_1e-05_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_161_mc_resnet_avg_epoch_200_lr_1e-05_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_160_mc_resnet_avg_epoch_200_lr_1e-05_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_160_mc_resnet_avg_epoch_200_lr_1e-05_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    # ]
    # label_list = ['lr = ' + r'$1\times10^{-5}$' + ', batch = 256, W/ shortcut',
    #               'lr = ' + r'$1\times10^{-5}$' + ', batch = 256, W/O shortcut']
    #
    # plot_error(file_path_list=file_path_list,
    #            label_list=label_list,
    #
    #            inter=751,
    #            inter_val=inter_val,
    #            y_ul=200,
    #            y_ll=0.5,
    #            x_ul=750 * 200,
    #            x_ll=0,
    #            y_log=y_log,
    #            x_log=False,
    #            x_scale=10000,
    #            save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/error_256_e_200_lr_00001.png'
    #            )
    #
    # file_path_list = [
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_159_mc_resnet_avg_epoch_200_lr_1e-05_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_159_mc_resnet_avg_epoch_200_lr_1e-05_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_158_mc_resnet_avg_epoch_200_lr_1e-05_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_158_mc_resnet_avg_epoch_200_lr_1e-05_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    # ]
    # label_list = ['lr = ' + r'$1\times10^{-5}$' + ', batch = 128, W/ shortcut',
    #               'lr = ' + r'$1\times10^{-5}$' + ', batch = 128, W/O shortcut']
    #
    # plot_error(file_path_list=file_path_list,
    #            label_list=label_list,
    #
    #            inter=1501,
    #            inter_val=inter_val,
    #            y_ul=100,
    #            y_ll=0.3,
    #            x_ul=1500 * 200,
    #            x_ll=0,
    #            y_log=y_log,
    #            x_log=False,
    #            x_scale=10000,
    #            save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/error_128_e_200_lr_00001.png'
    #            )
    #
    # file_path_list = [
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_157_mc_resnet_avg_epoch_200_lr_1e-05_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_157_mc_resnet_avg_epoch_200_lr_1e-05_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_156_mc_resnet_avg_epoch_200_lr_1e-05_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_156_mc_resnet_avg_epoch_200_lr_1e-05_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    # ]
    # label_list = ['lr = ' + r'$1\times10^{-5}$' + ', batch = 64, W/ shortcut',
    #               'lr = ' + r'$1\times10^{-5}$' + ', batch = 64, W/O shortcut']
    #
    # plot_error(file_path_list=file_path_list,
    #            label_list=label_list,
    #
    #            inter=3001,
    #            inter_val=inter_val,
    #            y_ul=y_ul,
    #            y_ll=0.3,
    #            x_ul=3000 * 200,
    #            x_ll=0,
    #            y_log=y_log,
    #            x_log=False,
    #            x_scale=10000,
    #            save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/error_64_e_200_lr_00001.png'
    #            )
    #
    # file_path_list = [
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_155_mc_resnet_avg_epoch_200_lr_1e-05_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_155_mc_resnet_avg_epoch_200_lr_1e-05_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_154_mc_resnet_avg_epoch_200_lr_1e-05_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_154_mc_resnet_avg_epoch_200_lr_1e-05_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    # ]
    # label_list = ['lr = ' + r'$1\times10^{-5}$' + ', batch = 32, W/ shortcut',
    #               'lr = ' + r'$1\times10^{-5}$' + ', batch = 32, W/O shortcut']
    #
    # plot_error(file_path_list=file_path_list,
    #            label_list=label_list,
    #
    #            inter=6001,
    #            inter_val=inter_val,
    #            y_ul=y_ul,
    #            y_ll=0.3,
    #            x_ul=6000 * 200,
    #            x_ll=0,
    #            y_log=y_log,
    #            x_log=False,
    #            x_scale=10000,
    #            save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/error_32_e_200_lr_00001.png'
    #            )
    #
    #
    #
    # file_path_list = [
    #
    #
    #     {'train':'/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_53_mc_resnet_avg_epoch_200_lr_0.0001_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #      'valid':'/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_53_mc_resnet_avg_epoch_200_lr_0.0001_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    #
    #
    #     {'train':'/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_52_mc_resnet_avg_epoch_200_lr_0.0001_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #      'valid':'/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_52_mc_resnet_avg_epoch_200_lr_0.0001_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    # ]
    # label_list = ['lr = ' + r'$1\times10^{-4}$' + ', batch = 256, W/ shortcut',
    #               'lr = ' + r'$1\times10^{-4}$' + ', batch = 256, W/O shortcut']
    #
    # plot_error(file_path_list=file_path_list,
    #            label_list=label_list,
    #
    #            inter=751,
    #            inter_val=inter_val,
    #            y_ul=y_ul,
    #            y_ll=0.2,
    #            x_ul=750*200,
    #            x_ll=0,
    #            y_log=y_log,
    #            x_log=False,
    #            x_scale=10000,
    #            save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/error_256_e_200_lr_0001.png'
    #            )
    #
    #
    # file_path_list = [
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_51_mc_resnet_avg_epoch_200_lr_0.0001_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_51_mc_resnet_avg_epoch_200_lr_0.0001_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_50_mc_resnet_avg_epoch_200_lr_0.0001_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_50_mc_resnet_avg_epoch_200_lr_0.0001_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    # ]
    # label_list = ['lr = ' + r'$1\times10^{-4}$' + ', batch = 128, W/ shortcut',
    #               'lr = ' + r'$1\times10^{-4}$' + ', batch = 128, W/O shortcut']
    #
    # plot_error(file_path_list=file_path_list,
    #            label_list=label_list,
    #
    #            inter=1501,
    #            inter_val=inter_val,
    #            y_ul=y_ul,
    #            y_ll=0.2,
    #            x_ul=1500*200,
    #            x_ll=0,
    #            y_log=y_log,
    #            x_log=False,
    #            x_scale=10000,
    #            save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/error_128_e_200_lr_0001.png'
    #            )
    #
    #
    #
    # file_path_list = [
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_111_mc_resnet_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_111'
    #                  '_mc_resnet_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_48_mc_resnet_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_48_mc_resnet_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    # ]
    # label_list = ['lr = ' + r'$1\times10^{-4}$' + ', batch = 64, W/ shortcut',
    #               'lr = ' + r'$1\times10^{-4}$' + ', batch = 64, W/O shortcut']
    #
    # plot_error(file_path_list=file_path_list,
    #            label_list=label_list,
    #
    #            inter=3001,
    #            inter_val=inter_val,
    #            y_ul=y_ul,
    #            y_ll=0.2,
    #            x_ul=3000 * 200,
    #            x_ll=0,
    #            y_log=y_log,
    #            x_log=False,
    #            x_scale=10000,
    #            save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/error_64_e_200_lr_0001.png'
    #            )
    #
    # file_path_list = [
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_62_mc_resnet_avg_epoch_200_lr_0.0001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_62_mc_resnet_avg_epoch_200_lr_0.0001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_162_mc_resnet_avg_epoch_200_lr_0.0001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_162_mc_resnet_avg_epoch_200_lr_0.0001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    # ]
    # label_list = ['lr = ' + r'$1\times10^{-4}$' + ', batch = 32, W/ shortcut',
    #               'lr = ' + r'$1\times10^{-4}$' + ', batch = 32, W/O shortcut']
    #
    # plot_error(file_path_list=file_path_list,
    #            label_list=label_list,
    #
    #            inter=6001,
    #            inter_val=inter_val,
    #            y_ul=y_ul,
    #            y_ll=0.1,
    #            x_ul=6000 * 200,
    #            x_ll=0,
    #            y_log=y_log,
    #            x_log=False,
    #            x_scale=10000,
    #            save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/error_32_e_200_lr_0001.png'
    #            )
    # #
    # file_path_list = [
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_59_mc_resnet_avg_epoch_200_lr_0.001_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_59_mc_resnet_avg_epoch_200_lr_0.001_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_58_mc_resnet_avg_epoch_200_lr_0.001_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_58_mc_resnet_avg_epoch_200_lr_0.001_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    # ]
    # label_list = ['lr = ' + r'$1\times10^{-3}$' + ', batch = 256, W/ shortcut',
    #               'lr = ' + r'$1\times10^{-3}$' + ', batch = 256, W/O shortcut']
    #
    # plot_error(file_path_list=file_path_list,
    #            label_list=label_list,
    #
    #            inter=751,
    #            inter_val=inter_val,
    #            y_ul=y_ul,
    #            y_ll=0.1,
    #            x_ul=750 * 200,
    #            x_ll=0,
    #            y_log=y_log,
    #            x_log=False,
    #            x_scale=10000,
    #            save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/error_256_e_200_lr_001.png'
    #            )
    #
    # file_path_list = [
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_93_mc_resnet_avg_epoch_200_lr_0.001_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_93_mc_resnet_avg_epoch_200_lr_0.001_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_56_mc_resnet_avg_epoch_200_lr_0.001_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_56_mc_resnet_avg_epoch_200_lr_0.001_batch_128_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    # ]
    # label_list = ['lr = ' + r'$1\times10^{-3}$' + ', batch = 128, W/ shortcut',
    #               'lr = ' + r'$1\times10^{-3}$' + ', batch = 128, W/O shortcut']
    #
    # plot_error(file_path_list=file_path_list,
    #            label_list=label_list,
    #
    #            inter=1501,
    #            inter_val=inter_val,
    #            y_ul=y_ul,
    #            y_ll=0.1,
    #            x_ul=1500 * 200,
    #            x_ll=0,
    #            y_log=y_log,
    #            x_log=False,
    #            x_scale=10000,
    #            save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/error_128_e_200_lr_001.png'
    #            )
    #
    file_path_list = [

        {
            'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_80_mc_resnet_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
            'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_80_mc_resnet_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},

        {
            'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_273_mc_resnet_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
            'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_273_mc_resnet_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},

    ]
    label_list = ['lr = ' + r'$1\times10^{-3}$' + ', batch = 64, W/ shortcut',
                  'lr = ' + r'$1\times10^{-3}$' + ', batch = 64, W/O shortcut']

    plot_error(file_path_list=file_path_list,
               label_list=label_list,

               inter=3001,
               inter_val=inter_val,
               y_ul=y_ul,
               y_ll=0.1,
               x_ul=3000 * 200,
               x_ll=0,
               y_log=y_log,
               x_log=False,
               x_scale=10000,
               save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/error_64_e_200_lr_001.png'
               )

    # file_path_list = [
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_61_mc_resnet_avg_epoch_200_lr_0.001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_61_mc_resnet_avg_epoch_200_lr_0.001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    #     {
    #         'train': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_60_mc_resnet_avg_epoch_200_lr_0.001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv',
    #         'valid': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_60_mc_resnet_avg_epoch_200_lr_0.001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_valid_200_1.csv'},
    #
    # ]
    # label_list = ['lr = ' + r'$1\times10^{-3}$' + ', batch = 32, W/ shortcut',
    #               'lr = ' + r'$1\times10^{-3}$' + ', batch = 32, W/O shortcut']
    #
    # plot_error(file_path_list=file_path_list,
    #            label_list=label_list,
    #
    #            inter=6001,
    #            inter_val=inter_val,
    #            y_ul=y_ul,
    #            y_ll=0.1,
    #            x_ul=6000 * 200,
    #            x_ll=0,
    #            y_log=y_log,
    #            x_log=False,
    #            x_scale=10000,
    #            save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/error_32_e_200_lr_001.png'
    #            )

    # TODO not good

    # for i in glob.glob('/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_*_mc_resnet_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv'):
    #     for j in glob.glob(
    #             '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_273_mc_resnet_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_0_fk_7_fs_2_fp_3_v1/loss_train_200_1.csv'):
    #
    #         valid_path_st1 = i.replace('train', 'valid')
    #         valid_path_st0 = j.replace('train', 'valid')
    #
    #         if os.path.exists(valid_path_st1) and os.path.exists(valid_path_st0):
    #             print(i)
    #             print(j)
    #             file_path_list = [
    #
    #                 {
    #                     'train': i,
    #                     'valid': valid_path_st1},
    #
    #                 {
    #                     'train': j,
    #                     'valid': valid_path_st0},
    #
    #             ]
    #             label_list = [list((list(i.split('/'))[-2]).split('_'))[1],
    #                           list((list(j.split('/'))[-2]).split('_'))[1]]
    #
    #             plot_error(file_path_list=file_path_list,
    #                        label_list=label_list,
    #
    #                        inter=3001,
    #                        inter_val=inter_val,
    #                        y_ul=50,
    #                        y_ll=0.1,
    #                        x_ul=3000 * 200,
    #                        x_ll=0,
    #                        y_log=True,
    #                        x_log=False,
    #                        x_scale=10000,
    #                        save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/test'
    #                        )

    pass
