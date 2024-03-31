#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/24 23:50
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : make_dataset.py
# @Software: PyCharm

# import sys
# sys.path.append('../')

import pandas as pd
from read_root import ReadRoot
import numpy as np
import uproot
import os
from collections import Counter
import random

# def convert_root(file_path,
#                  tree_name, exp_dict, threshold, e_random_scale, e_scale=1, save_path=None, start=None,
#                  end=None):
#
#     exp = list(exp_dict.keys())
#
#     root_file = ReadRoot(file_path=file_path, tree_name=tree_name, exp=exp, start=start, end=end)
#
#
#     # read raw root file
#
#     num_events = len(e)
#     assert num_events == len(x)
#     assert num_events == len(y)
#     assert num_events == len(z)
#
#     # NHWC
#     deposits = []
#
#     for i in range(num_events):
#
#         deposit_ = np.zeros((1, 18, 18, 40))
#
#         energies_ = e[i]
#
#         x_ = np.around((x[i] + 342.549) / 40.3).astype(int)
#         y_ = np.around((y[i] + 342.549) / 40.3).astype(int)
#         z_ = ((z[i]) / 30).astype(int)
#         num_events_ = len(energies_)
#         assert num_events_ == len(x_)
#         assert num_events_ == len(y_)
#         assert num_events_ == len(z_)
#
#         for j in range(num_events_):
#             energy = energies_[j] if energies_[j] > threshold else 0
#             deposit_[0, x_[j], y_[j], z_[j]] += energy
#
#         if e_random_scale:
#             deposit_ = deposit_ * (0.5 + np.random.rand(1))
#         deposits.append(deposit_)
#
#     deposits = np.concatenate(deposits)
#     deposits = deposits * e_scale
#
#     if save_path != None:
#         np.save(save_path, deposits)
#         return
#     return deposits
#
#
# def merge(file_lists, save_path, tree_name, exps, threshold, start, end, e_random_scale, **kwargs):
#     '''file_lists: the root file to be converted'''
#
#     if 'e_scale_list' in kwargs.keys():
#         result = [
#             convert_root(file_path=file_path, exps=exps, threshold=threshold, tree_name=tree_name, start=start, end=end,
#                          e_random_scale=e_random_scale, e_scale=e_scale)
#             for file_path, e_scale in zip(file_lists, kwargs.get('e_scale_list'))]
#     else:
#         result = [
#             convert_root(file_path=file_path, exps=exps, threshold=threshold, tree_name=tree_name, start=start, end=end,
#                          e_random_scale=e_random_scale) for file_path in file_lists]
#     result = np.concatenate(result, axis=0)
#     np.save(save_path, result)
#
# def prepare_labels(file_dict, save_dir):
#     for key, value in file_dict.items():
#         labels = key * np.ones(len(np.load(value)))
#         save_path = os.path.join(save_dir, '{}.npy'.format(key))
#         np.save(save_path, labels)
#
#
# def merge_npy_dataset(img_file_list, label_file_list, save_dir, statistics_per_file=-1):
#     imgs = []
#     labels = []
#
#     for img_path, label_path in zip(img_file_list, label_file_list):
#
#         img = np.load(img_path)
#         label = np.load(label_path)
#
#         len_img_ = len(img)
#         len_label_ = len(label)
#         assert len_img_ == len_label_
#
#         if statistics_per_file > 0:
#             if len_img_ >= statistics_per_file:
#                 choice = np.random.choice(np.arange(len_img_), statistics_per_file, replace=False)
#             else:
#                 choice = np.random.choice(np.arange(len_img_), statistics_per_file, replace=True)
#
#             img = img[choice]
#             label = label[choice]
#
#         imgs.append(img)
#         labels.append(label)
#
#     imgs = np.concatenate(imgs)
#     labels = np.concatenate(labels)
#
#     assert len(imgs) == len(labels)
#     if save_dir != None:
#         np.save(os.path.join(save_dir, 'imgs.npy'), imgs)
#         np.save(os.path.join(save_dir, 'labels.npy'), labels)
#         return
#     else:
#         return imgs, labels
#
#
# def make_final_dataset(img_file_list, label_file_list, save_dir, ratio, statistics_per_file=-1):
#     '''it makes datasets from .npy seperated in different particel types'''
#
#     imgs, labels = merge_npy_dataset(img_file_list=img_file_list, label_file_list=label_file_list, save_dir=None,
#                                      statistics_per_file=statistics_per_file)
#     length = len(labels)
#     indexes = np.arange(length).astype(np.longlong)
#
#     np.random.shuffle(indexes)
#
#     dir_dict = {
#         'Train': indexes[:int(length * (ratio[0] / sum(ratio)))],
#         'Validation': indexes[int(length * (ratio[0] / sum(ratio))):int(length * ((ratio[1] + ratio[0]) / sum(ratio)))],
#         'Test': indexes[int(length * ((ratio[1] + ratio[0]) / sum(ratio))):],
#     }
#
#     for key, value in dir_dict.items():
#         save_dir_ = os.path.join(save_dir, key)
#         if not os.path.exists(save_dir_):
#             os.mkdir(save_dir_)
#
#         labels_ = labels[value]
#
#
#
#         np.save(os.path.join(save_dir_, 'imgs.npy'), imgs[value])
#         np.save(os.path.join(save_dir_, 'labels.npy'), labels_)
#
#         counter = dict(Counter(labels_))
#         filename = open(os.path.join(save_dir_, 'log.txt'), 'w')  # dict to txt
#         for k, v in (counter).items():
#             filename.write(str(k) + ':' + str(v))
#             filename.write('\n')
#         filename.close()
#
#
#
# def mix_event(file_1,
#               label_1,
#               file_2,
#               label_2,
#               length_x,
#               length_y,
#               trans_x1,
#               trans_y1,
#               trans_x2,
#               trans_y2,
#               threshold):
#     def transform(index, trans_x, trans_y, ):
#         index[0] = index[0] + trans_x
#         index[1] = index[1] + trans_y
#
#         index_x = np.logical_and(index[0] >= 0, index[0] < length_x)
#         index_y = np.logical_and(index[1] >= 0, index[1] < length_y)
#
#         index_trans = np.logical_and(index_x, index_y)
#
#         for i in range(3):
#             index[i] = index[i][index_trans]
#
#     index_1 = list(np.where(file_1 >= threshold))
#     index_2 = list(np.where(file_2 >= threshold))
#
#
#
#     # transform
#     transform(index_1, trans_x=trans_x1, trans_y=trans_y1)
#     transform(index_2, trans_x=trans_x2, trans_y=trans_y2)
#
#     e_1 = file_1[index_1[0]-trans_x1, index_1[1]-trans_y1, index_1[2]]
#     e_2 = file_2[index_2[0]-trans_x2, index_2[1]-trans_y2, index_2[2]]
#
#     index_1.append(e_1)
#     index_1.append(label_1 * np.ones(len(e_1)))
#
#     index_2.append(e_2)
#     index_2.append(label_2 * np.ones(len(e_2)))
#
#     return np.hstack([np.vstack(index_1), np.vstack(index_2)]).tolist()
#
#
# def make_data(imgs_1,
#               labels_1,
#               imgs_2,
#               labels_2,
#               length_x,
#               length_y,
#               trans_x1,
#               trans_y1,
#               trans_x2,
#               trans_y2,
#               threshold):
#     results = []
#
#     for file_1, label_1, file_2, label_2 in zip(imgs_1, labels_1, imgs_2, labels_2):
#         results.append(mix_event(file_1=file_1,
#                                  file_2=file_2,
#                                  label_1=label_1,
#                                  label_2=label_2,
#                                  length_x=length_x,
#                                  length_y=length_y,
#                                  trans_x1=trans_x1,
#                                  trans_y1=trans_y1,
#                                  trans_x2=trans_x2,
#                                  trans_y2=trans_y2,
#                                  threshold=threshold
#                                  ))
#
#     return results
#
#
# def main_make_data(file_path,
#                    label_path,
#                    length_x,
#                    length_y,
#                    trans_x1,
#                    trans_y1,
#                    trans_x2,
#                    trans_y2,
#                    threshold,
#                    save_dir):
#     imgs = np.load(file_path)
#     labels = np.load(label_path)
#
#     muon = imgs[labels == 0]
#     electron = imgs[labels == 1]
#     pion = imgs[labels == 2]
#
#     assert len(muon) == len(electron)
#     assert len(muon) == len(pion)
#
#
#
#     mu_e = make_data(imgs_1=muon, labels_1=labels[labels == 0],
#                       imgs_2=electron, labels_2=labels[labels == 1],
#                       length_x=length_x,
#                       length_y=length_y,
#                       trans_x1=trans_x1,
#                       trans_y1=trans_y1,
#                       trans_x2=trans_x2,
#                       trans_y2=trans_y2,
#                       threshold=threshold
#                       )
#
#
#     mu_pi = make_data(imgs_1=muon, labels_1=labels[labels == 0],
#                       imgs_2=pion, labels_2=labels[labels == 2],
#                       length_x=length_x,
#                       length_y=length_y,
#                       trans_x1=trans_x1,
#                       trans_y1=trans_y1,
#                       trans_x2=trans_x2,
#                       trans_y2=trans_y2,
#                       threshold=threshold
#                       )
#
#
#
#     e_pi = make_data(imgs_1=electron, labels_1=labels[labels == 1],
#                       imgs_2=pion, labels_2=labels[labels == 2],
#                       length_x=length_x,
#                       length_y=length_y,
#                       trans_x1=trans_x1,
#                       trans_y1=trans_y1,
#                       trans_x2=trans_x2,
#                       trans_y2=trans_y2,
#                       threshold=threshold
#                       )
#
#
#     result=mu_e + mu_pi + e_pi
#
#
#     total_n=len(result)
#
#     # print(result[1:3])
#     random.shuffle(result)
#
#     p_70 = int(0.7 * total_n)
#     p_90 = int(0.9 * total_n)
#
#     os.makedirs(save_dir, exist_ok=True)
#
#
#     os.makedirs(os.path.join(save_dir, 'Train'), exist_ok=True)
#     np.save(os.path.join(save_dir, 'Train/points.npy'), result[:p_70])
#
#     os.makedirs(os.path.join(save_dir, 'Validation'), exist_ok=True)
#     np.save(os.path.join(save_dir, 'Validation/points.npy'), result[p_70:p_90])
#
#     os.makedirs(os.path.join(save_dir, 'Test'), exist_ok=True)
#     np.save(os.path.join(save_dir, 'Test/points.npy'), result[p_90:])


def main_debug():
    os.makedirs('/lustre/collider/songsiyuan/TriHiggs/dataset', exist_ok=True)
    os.makedirs('/lustre/collider/songsiyuan/TriHiggs/dataset/debug', exist_ok=True)

    exp = ['njets', 'nbjets', 'TotalJetsEnergy', 'TotalHiggsJetPt', 'mH1', 'mH2', 'mH3', 'phiH1', 'phiH2', 'phiH3',
           'dEtaH1', 'dEtaH2', 'dEtaH3', 'dPhiH1', 'dPhiH2', 'dPhiH3', 'dRH1', 'dRH2', 'dRH3', 'pTH1', 'pTH2', 'pTH3',
           'mHangle1', 'mHangle2', 'mHangle3', 'asymH1', 'asymH2', 'asymH3', 'circH1', 'circH2', 'circH3', 'mHHH',
           'rmsmH', 'rmsOverMeanH', 'cosAngleH1', 'cosAngleH2', 'cosAngleH3', 'eta_mHHH', 'isSignal']

    file = ReadRoot(file_path='/lustre/collider/zhoubaihong/tri-Higgs/ML/Signal_nano.root',
                    tree_name='HHHNtuple',
                    exp=exp)

    file.build_csv(branch_list=exp, save_path='/lustre/collider/songsiyuan/TriHiggs/dataset/debug/Signal_nano.csv')

    file = ReadRoot(file_path='/lustre/collider/zhoubaihong/tri-Higgs/ML/Background_nano.root',
                    tree_name='HHHNtuple',
                    exp=exp)

    file.build_csv(branch_list=exp, save_path='/lustre/collider/songsiyuan/TriHiggs/dataset/debug/Background_nano.csv')

    # #combiine
    sig = pd.read_csv('/lustre/collider/songsiyuan/TriHiggs/dataset/debug/Signal_nano.csv')
    bkg = pd.read_csv('/lustre/collider/songsiyuan/TriHiggs/dataset/debug/Background_nano.csv')

    sig_bkg = pd.concat([sig, bkg])

    sig_bkg.to_csv('/lustre/collider/songsiyuan/TriHiggs/dataset/debug/sig_bkg_nano.csv')

if __name__ == '__main__':


    pass
