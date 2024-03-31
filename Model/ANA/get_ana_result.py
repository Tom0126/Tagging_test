#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/1 23:40
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : get_auc_result.py
# @Software: PyCharm

import glob
import math

import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
pd.set_option('display.max_rows', None)
import os
from Evaluate import calculate_auc
from ANA.ann_ana import calculate_acc
def main(file_format, **kwargs):
    auc_dict = dict()
    for path in glob.glob(file_format):
        df = pd.read_csv(path)
        if 'effi_points' in kwargs.keys():
            df=df[df['Unnamed: 0'].isin(kwargs.get('effi_points'))]
        print(path, '\n', df)

        ann_bkg_ra = df['ann_bkg_ra'].values

        auc_dict[list(path.split('/'))[-3]] = ann_bkg_ra[-5]

    # auc_dict = dict(sorted(auc_dict.items(), key=lambda x: x[1], reverse=True))
    #
    # for key, value in auc_dict.items():
    #     print(key, value)


def main_by_sort(file_format, split_key, effi_points, **kwargs):

    file_dict = {}

    file_list = glob.glob(file_format)

    for file in file_list:
        file_name = list(file.split('/'))[-1]
        file_name = list(file_name.split(split_key))[-1]

        if file_name in file_dict.keys():
            file_dict[file_name].append(file)

        else:
            file_dict[file_name] = [file]

    ana_dict = {}


    for file_name, file_list in file_dict.items():
        show_index = None
        effi_=None
        ra_ = list()
        auc_ = list()
        acc_=list()

        ll_list = kwargs.get('ll', [0])
        ul_list = kwargs.get('ul', [999999999])


        for path in file_list:

            ra_path_=os.path.join(path, 'ANA/ann_info_s_1_b_0.csv')
            auc_path_ = os.path.join(path, 'ANA/roc/auroc.npy')
            ann_score_path_=os.path.join(path, 'ANA/imgs_ANN.csv')
            # ra_detailed_path_=os.path.join(path, 'ANA/ann_info_s_1_b_0_detailed.csv')

            if os.path.exists(ra_path_) and os.path.exists(auc_path_) and os.path.exists(ann_score_path_):
                df = pd.read_csv(ra_path_)

                if set(effi_points).issubset(set(list(df['Unnamed: 0']))) and len(df) == kwargs.get('length', 0):

                    ra_index_list = [np.where(df['Unnamed: 0'].values == i_)[0][0] for i_ in
                                     kwargs.get('target_effi', [0.95])]
                    ann_bkg_ra = df['ann_bkg_ra'].values

                    auc = np.load(auc_path_)

                    flag=True
                    for ll, ul, ra_index in zip(ll_list, ul_list, ra_index_list):
                         flag = flag and ann_bkg_ra[ra_index] >= ll and ann_bkg_ra[ra_index] <= ul
                    if flag:
                        ra_.append(ann_bkg_ra.reshape(-1, 1))
                        auc_.append(auc[1])
                        acc_.append(calculate_acc(ann_score_path_))

                    show_index=df['Unnamed: 0'].isin(effi_points)
                    effi_=df['Unnamed: 0']


        auc_ = np.array(auc_)
        acc_ = np.array(acc_)

        if len(ra_) > 0:
            ra_ = np.hstack(ra_)

            fixed_num= kwargs.get('fixed_num',ra_.shape[-1] )


            ra_=ra_[:,:fixed_num]
            auc_=auc_[:fixed_num]
            acc_=acc_[:fixed_num]

            ra_mean = np.mean(ra_, axis=1)
            ra_std = np.std(ra_, axis=1)

            auc_mean = np.mean(auc_)
            auc_std = np.std(auc_)

            acc_mean = np.mean(acc_)
            acc_std = np.std(acc_)

            ana_dict[file_name] = {'ra_mean': ra_mean,
                                   'ra_std': ra_std / math.sqrt(fixed_num),
                                   'num': fixed_num,
                                   'auc_mean': auc_mean,
                                   'auc_std': auc_std/ math.sqrt(fixed_num),
                                   'acc_mean': acc_mean,
                                   'acc_std': acc_std/ math.sqrt(fixed_num),
                                   'show_index':show_index,
                                   'ana_effi':effi_
                                   }



    ana_dict = dict(sorted(ana_dict.items(), key=lambda x: x[1]['ra_mean'][0], reverse=True))

    for file_name, ana_result in ana_dict.items():

        print(file_name, ana_result['num'],
              '{:.2f} + {:.2f}'.format(ana_result['acc_mean']*100, ana_result['acc_std']*100),
              ['{} + {}'.format(round(mean, 0), round(std, 0)) for mean, std in
               zip(ana_result['ra_mean'][ana_result['show_index'].values], ana_result['ra_std'][ana_result['show_index'].values])])
        df_=pd.DataFrame({'ann_effi': ana_result['ana_effi'],
                          'ann_bkg_ra': ana_result['ra_mean']})



        df_.to_csv(os.path.join('/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result', file_name+'.csv'))

if __name__ == '__main__':
    # effi_points= [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]
    # effi_points = [ 0.97, 0.98, 0.99, 0.995]
    effi_points = [0.95,  0.96, 0.97,  0.98, 0.99, 0.995]
    # effi_points= np.linspace(0.95, 0.995, 100)

    # main('/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0116*/ANA/ann_info_s_1_b_0.csv')
    main_by_sort(
        file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0116*',
        split_key='_mc_',
        ul=[8500, 450],
        ll=[0, 0],
        target_effi=[0.98, 0.995],
        length=111,
        effi_points=effi_points
    )
    # for i in [8000, 8500]:
    #     for j in range(400, 600, 50):
    #         print(i, j)
    #         main_by_sort(
    #             file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0116*',
    #             split_key='_mc_',
    #             ul=[i, j],
    #             ll=[0, 0],
    #             target_effi=[0.98, 0.995],
    #             length=111,
    #             effi_points=effi_points
    #         )
    # # main_by_sort(
    # #     file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0116*',
    # #     split_key='_mc_',
    # #     # ul=9000,
    # #     # target_effi=0.98,
    # #     length=40,
    # #     effi_points=effi_points
    # # )
    print('\n')
    # # main('/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0117*/ANA/ann_info_s_1_b_0.csv')
    # # main_by_sort(
    # #     file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0117*',
    # #     split_key='_mc_',
    # #     ul=2500,
    # #     ra_index=-2,
    # #     effi_points=effi_points
    # # )
    # # print('\n')
    # # main('/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0118*/ANA/ann_info_s_1_b_0.csv')
    #
    main_by_sort(
        file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0118*',
        split_key='_mc_',
        ul=[1600],
        ll=[700],
        length=111,
        target_effi=[0.99],
        effi_points=effi_points
           )
    # #
    # # main_by_sort(
    # #     file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0118*',
    # #     split_key='_mc_',
    # #     length=40,
    # #
    # #     effi_points=effi_points
    # # )
    # #
    print('\n')
    # #
    main('/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120*/ANA/ann_info_s_1_b_0.csv',
         effi_points=effi_points)
    # main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120*',
    #              split_key='_mc_',
    #              # cut_value=50000,
    #
    #              length=5,
    #
    #              effi_points=effi_points) #cut_value 50000
    #
    # main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120*',
    #              split_key='_mc_',
    #              # cut_value=50000,
    #
    #              length=100,
    #
    #              effi_points=effi_points)
    main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120*',
                 split_key='_mc_',
                 length=111,

                 effi_points=effi_points)
    # # main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120*',
    # #              split_key='_mc_',
    # #              # cut_value=50000,
    # #              ll=1600,
    # #              length=40,
    # #              target_effi=0.99,
    # #              effi_points=effi_points) #cut_value 50000
    #
    #
    #
    # # main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120*',
    # #              split_key='_mc_',
    # #              # cut_value=50000,
    # #              target_effi=0.97,
    # #              ul=25000,
    # #              ll=14000,
    # #              effi_points=effi_points)  # cut_value 50000
    #
    #
    #
    print('\n')
    main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120*',
                 split_key='_mc_',
                 # cut_value=50000,
                 target_effi=[0.98, 0.97,],
                 ul=[12000, 35000 - 3000, ],
                 ll=[6400, 10000, ],
                 length=111,
                 effi_points=effi_points)  # cut_value 50000

    # main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120*',
    #              split_key='_mc_',
    #              # cut_value=50000,
    #              # target_effi=0.97,
    #              # ul=31000,
    #              length=40,
    #              effi_points=effi_points)  # cut_value 50000

    print('\n')

    main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120*resnet_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_0.1_step_10_st_1_fk_3_fs_1_fp_1_v1',
                 split_key='_mc_',
                 length=111,

                 effi_points=effi_points)

    print('\n')



    # # main('/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/*mc_dgcnn_epoch_30_lr_0.1_batch_128_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_128_k_20_v1/ANA/ann_info_s_1_b_0.csv')
    # main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119*',
    #              split_key='_mc_',
    #              # cut_value=0,
    #              # fixed_num=10,
    #              length=111,
    #              effi_points=effi_points)

    # for i in range(50000, 80000, 2000):
    #     print(i)
    #     main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119*',
    #                  split_key='_mc_',
    #                  # cut_value=0,
    #                  # fixed_num=10,
    #                  ul=[i],
    #                  target_effi=[0.95],
    #                  length=111,
    #                  effi_points=effi_points)

    main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119*',
                 split_key='_mc_',
                 # cut_value=0,
                 # fixed_num=10,
                 ul=[64000],
                 target_effi=[0.95],
                 length=111,
                 effi_points=effi_points)
    # #
    print('\n')
    #
    # # main('/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121*/ANA/ann_info_s_1_b_0.csv')
    # main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121*',
    #              split_key='_mc_',
    #              # cut_value=45000,
    #              ll=[20000],
    #              ul=[50000],
    #              # ul=700,
    #              length=111,
    #              target_effi=[0.95],
    #              effi_points=effi_points)
    #
    # for i in range(45000, 55000, 2000):
    #     print(i)
    #     main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121*',
    #                  split_key='_mc_',
    #                  # cut_value=45000,
    #                  ll=[i],
    #                  length=111,
    #                  target_effi=[0.95],
    #                  effi_points=effi_points)

    main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121*',
                 split_key='_mc_',
                 # cut_value=45000,
                 ll=[47000],
                 length=111,
                 target_effi=[0.95],
                 effi_points=effi_points)
    # print('\n')
    #
    # # main('/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0122*/ANA/ann_info_s_1_b_0.csv')


    main_by_sort(file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0122*',
                 split_key='_mc_',
                 ul=[5000],
                 length=111,
                 target_effi=[0.99],
                 effi_points=effi_points)

    #grav 33, 35, 78, 73, 79 cut



    pass
