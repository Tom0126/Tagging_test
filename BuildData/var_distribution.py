#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/21 15:48
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : var_distribution.py
# @Software: PyCharm

import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from BuildData.read_root import ReadRoot

def main_validation_var():
    os.makedirs('/lustre/collider/songsiyuan/TriHiggs/dataset', exist_ok=True)
    os.makedirs('/lustre/collider/songsiyuan/TriHiggs/dataset/validation', exist_ok=True)

    exp = ['njets', 'nbjets', 'TotalJetsEnergy', 'TotalHiggsJetPt', 'mH1', 'mH2', 'mH3', 'phiH1', 'phiH2', 'phiH3',
           'dEtaH1', 'dEtaH2', 'dEtaH3', 'dPhiH1', 'dPhiH2', 'dPhiH3', 'dRH1', 'dRH2', 'dRH3', 'pTH1', 'pTH2', 'pTH3',
           'mHangle1', 'mHangle2', 'mHangle3', 'asymH1', 'asymH2', 'asymH3', 'circH1', 'circH2', 'circH3', 'mHHH',
           'rmsmH', 'rmsOverMeanH', 'cosAngleH1', 'cosAngleH2', 'cosAngleH3', 'eta_mHHH', 'isSignal']

    file = ReadRoot(file_path='/lustre/collider/wanghaoyu/Ntuples/triHiggs_ML_v1/validation/triHiggs_ML.root',
                    tree_name='HHHNtuple',
                    exp=exp)

    file.build_csv(branch_list=exp, save_path='/lustre/collider/songsiyuan/TriHiggs/dataset/validation/vector.csv')




def plot_distribution(file_path, cols, label_col, save_path):
    n_plots = len(cols)

    n_col = int(math.sqrt(n_plots))

    n_row = n_col

    while n_row * n_row < n_plots:
        n_row += 1

    vars = pd.read_csv(file_path, usecols=cols)

    fig = plt.figure(figsize=(6 * n_col, 5 * n_row))

    for i, col in enumerate(cols, 1):
        plt.subplot(n_row, n_col, i)
        mean_=np.mean(vars[col][vars[label_col] == 1])
        std_ = np.std(vars[col][vars[label_col] == 1])
        plt.hist(vars[col][vars[label_col] == 1],
                 label='signal\nAvg:{:.1e}\nStd:{:.1e}'.format(mean_,
                                                       std_),
                 histtype='stepfilled',
                 alpha=0.8,
                 density=True,
                 range=[mean_-3*std_, mean_+3*std_],
                 bins=20,
                 )

        mean_ = np.mean(vars[col][vars[label_col] == 0])
        std_ = np.std(vars[col][vars[label_col] == 0])

        plt.hist(vars[col][vars[label_col] == 0],


                 label='bkg\nAvg:{:.1e}\nStd:{:.1e}'.format(mean_,
                                                    std_),
                 histtype='stepfilled',
                 alpha=0.8,
                 density=True,
                 range=[mean_ - 3 * std_, mean_ + 3 * std_],
                 bins=20,
                 )

        plt.legend()

        plt.xlabel(col)

    plt.savefig(save_path)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':

    # main_validation_var()
    var_list = ['njets', 'nbjets', 'TotalJetsEnergy', 'TotalHiggsJetPt', 'mH1', 'mH2', 'mH3', 'phiH1', 'phiH2', 'phiH3',
                'dEtaH1', 'dEtaH2', 'dEtaH3', 'dPhiH1', 'dPhiH2', 'dPhiH3', 'dRH1', 'dRH2', 'dRH3', 'pTH1', 'pTH2',
                'pTH3', 'mHangle1', 'mHangle2', 'mHangle3', 'asymH1', 'asymH2', 'asymH3', 'circH1', 'circH2', 'circH3',
                'mHHH', 'rmsmH', 'rmsOverMeanH', 'cosAngleH1', 'cosAngleH2', 'cosAngleH3', 'eta_mHHH', 'isSignal']

    plot_distribution(file_path='/lustre/collider/songsiyuan/TriHiggs/dataset/validation/vector.csv',
                      cols=var_list,
                      label_col='isSignal',
                      save_path='/lustre/collider/songsiyuan/TriHiggs/dataset/debug/vat.png')


    # file = pd.read_csv('/lustre/collider/songsiyuan/TriHiggs/dataset/training/vector.csv',
    #                    usecols=['isSignal'])
    #
    # print('bkg {}'.format(np.sum((file['isSignal']==0)!=0)))
    # print('sig {}'.format(np.sum((file['isSignal'] == 1) != 0)))



    pass
