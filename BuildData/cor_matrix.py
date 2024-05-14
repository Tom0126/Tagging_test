#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 19:06
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : Cor.py
# @Software: PyCharm

from BuildData.read_root import ReadRoot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plotCorHeatmap_root(file_path, tree_name, var_list, save_path):
    data = ReadRoot(file_path=file_path, tree_name=tree_name, exp=var_list)
    data_dict = {}
    for var in var_list:
        data_dict[var] = data.readBranch(var)
    data_dict = pd.DataFrame(data_dict)
    df_coor = data_dict.corr()

    plt.subplots(figsize=(9, 9), facecolor='w')

    fig = sns.heatmap(df_coor, annot=True, vmax=1, square=True, cmap="PiYG",
                      fmt='.2g')
    fig.get_figure().savefig(save_path, bbox_inches='tight', transparent=True)


def plotCorHeatmap_csv(file_path, varlist, save_path, title, **kwargs):
    df = pd.read_csv(file_path, usecols=varlist)
    df_coor = df.corr()
    df.fillna(0)
    plt.subplots(figsize=(25, 15), facecolor='w')

    if 'labels' in kwargs:
        labels = kwargs.get('labels')
    else:
        labels = varlist

    fig = sns.heatmap(df_coor,
                      annot=kwargs.get('annot',True),
                      vmax=1, vmin=-1, square=True, cmap='RdBu',
                      fmt='.2f', annot_kws={"fontsize": 25, }, xticklabels=labels, yticklabels=labels)

    cbar = fig.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    # fig.set_title(title, fontsize=30)
    plt.tick_params(axis='x', labelsize=13)
    plt.tick_params(axis='y', labelsize=20)

    plt.xticks(rotation=90)
    fig.get_figure().savefig(save_path, bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == '__main__':



    #


    # var_list = ['njets', 'nbjets', 'TotalJetsEnergy', 'TotalHiggsJetPt', 'mH1', 'mH2', 'mH3', 'phiH1', 'phiH2', 'phiH3',
    #        'dEtaH1', 'dEtaH2', 'dEtaH3', 'dPhiH1', 'dPhiH2', 'dPhiH3', 'dRH1', 'dRH2', 'dRH3', 'pTH1', 'pTH2', 'pTH3',
    #        'mHangle1', 'mHangle2', 'mHangle3', 'asymH1', 'asymH2', 'asymH3', 'circH1', 'circH2', 'circH3', 'mHHH',
    #        'rmsmH', 'rmsOverMeanH', 'cosAngleH1', 'cosAngleH2', 'cosAngleH3', 'eta_mHHH']

    var_list = [ 'nbjets', 'mH1', 'mH2', 'mH3',
                'dPhiH1', 'dPhiH2', 'dPhiH3', 'dRH1', 'dRH2', 'dRH3', 'pTH1', 'pTH2',
                'pTH3',   'circH1', 'circH2', 'circH3',
                'mHHH', 'rmsmH', 'rmsOverMeanH', 'cosAngleH2', 'cosAngleH3', 'eta_mHHH']

    labels = var_list

    plotCorHeatmap_csv(
        file_path='/lustre/collider/songsiyuan/TriHiggs/dataset/training/vector.csv',
        varlist=var_list,
        save_path='/lustre/collider/songsiyuan/TriHiggs/dataset/training/coor_mc.png',
        title='MC samples',
        labels=labels,
        annot=False
    )


    pass
