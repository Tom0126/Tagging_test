#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/25 13:06
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : ann_ana.py
# @Software: PyCharm


import numpy as np
import pandas as pd
import math
import os
import uproot
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from sklearn.metrics import confusion_matrix
import math
from ANA.roc import calculate_auc


class ReadRoot():

    def __init__(self, file_path, tree_name, start=None, end=None, cut=None, exp=None):
        file = uproot.open(file_path)
        tree = file[tree_name]

        self.tree = tree.arrays(aliases=None, cut=cut, expressions=exp, library="np", entry_start=start,
                                entry_stop=end)

    def readBranch(self, branch):
        return self.tree[branch]


def read_ann_score(file_pid_path, n_classes=4, rt_df=False):
    branch_list_dict = {
        2: ['ANN_e_plus', 'ANN_pi_plus'],
        3: ['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', ],
        4: ['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise'],
    }
    branch_list = branch_list_dict.get(n_classes)

    ann_pid = ReadRoot(file_path=file_pid_path, tree_name='Calib_Hit', exp=branch_list)

    ann_score = {}
    for branch in branch_list:
        ann_score[branch] = ann_pid.readBranch(branch)

    if rt_df:
        return pd.DataFrame(ann_score)
    else:
        return pd.DataFrame(ann_score).values


class ANN_ANA():

    def __init__(self, ann_scores_path,
                 raw_labels_path,
                 save_dir,
                 ann_threshold_lists,
                 ann_signal_label,
                 n_classes,
                 **kwargs):

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.ann_scores = read_ann_score(ann_scores_path, n_classes,
                                         rt_df=False) if ann_scores_path != None else kwargs.get('ann_scores')
        self.raw_labels = np.load(raw_labels_path) if raw_labels_path != None else kwargs.get(
            'raw_labels')

        self.ann_threshold_lists = ann_threshold_lists
        self.ann_signal_label = ann_signal_label

        self.label_list = []

        self.n_classes = n_classes

    def filter_label(self, label_list):

        self.label_list = label_list
        ann_cut = self.raw_labels == label_list[0]

        for label in label_list:
            ann_cut = np.logical_or(ann_cut, self.raw_labels == label)

        self.ann_scores = self.ann_scores[ann_cut]
        self.raw_labels = self.raw_labels[ann_cut]

    def get_ann_purity(self):

        signal_scores = self.ann_scores[:, self.ann_signal_label]

        purities = []

        for threshold in self.ann_threshold_lists:
            signal_picked = self.raw_labels[signal_scores >= threshold]

            purities.append(np.sum((signal_picked == self.ann_signal_label) != 0) / len(signal_picked))

        return np.array(purities)

    def get_ann_efficiency(self):

        signal_scores = self.ann_scores[:, self.ann_signal_label]

        efficiencies = []

        for threshold in self.ann_threshold_lists:
            signal_picked = self.raw_labels[signal_scores >= threshold]

            efficiencies.append(
                np.sum((signal_picked == self.ann_signal_label) != 0) /
                np.sum((self.raw_labels == self.ann_signal_label) != 0))

        return np.array(efficiencies)

    def get_ann_bkg_ratio(self):

        signal_scores = self.ann_scores[:, self.ann_signal_label]

        bkg_ratios = []

        for threshold in self.ann_threshold_lists:
            signal_picked = self.raw_labels[signal_scores >= threshold]
            bkg_picked_num = np.sum((signal_picked != self.ann_signal_label) != 0)

            if bkg_picked_num > 0:
                bkg_ratios.append(np.sum((self.raw_labels != self.ann_signal_label) != 0) / bkg_picked_num)
            else:
                bkg_ratios.append(-1)

        return np.array(bkg_ratios)

    def get_ann_bkg_rate(self):

        signal_scores = self.ann_scores[:, self.ann_signal_label]

        bkg_rates = []

        for threshold in self.ann_threshold_lists:
            signal_picked = self.raw_labels[signal_scores >= threshold]
            bkg_picked_num = np.sum((signal_picked != self.ann_signal_label) != 0)

            bkg_rates.append(1 - bkg_picked_num / np.sum((self.raw_labels != self.ann_signal_label) != 0))

        return np.array(bkg_rates)

    def get_ann_bkg_efficiency(self):

        signal_scores = self.ann_scores[:, self.ann_signal_label]

        bkg_efficiencies = []

        for threshold in self.ann_threshold_lists:
            signal_picked = self.raw_labels[signal_scores >= threshold]
            bkg_picked_num = np.sum((signal_picked != self.ann_signal_label) != 0)

            bkg_efficiencies.append(bkg_picked_num / np.sum((self.raw_labels != self.ann_signal_label) != 0))

        return np.array(bkg_efficiencies)

    def get_ann_significance(self):

        signal_scores = self.ann_scores[:, self.ann_signal_label]

        significances = []

        for threshold in self.ann_threshold_lists:

            signal_picked = self.raw_labels[signal_scores >= threshold]
            bkg_picked_num = np.sum((signal_picked != self.ann_signal_label) != 0)

            if bkg_picked_num > 0:
                significances.append(np.sum((signal_picked == self.ann_signal_label) != 0) / math.sqrt(bkg_picked_num))
            else:
                significances.append(-1)

        return np.array(significances)

    def get_effi_bkg_ratio(self):

        signal_scores = self.ann_scores[:, self.ann_signal_label]

        efficiencies = []
        bkg_ratios = []

        for threshold in self.ann_threshold_lists:
            signal_picked = self.raw_labels[signal_scores >= threshold]

            efficiencies.append(
                np.sum((signal_picked == self.ann_signal_label) != 0) /
                np.sum((self.raw_labels == self.ann_signal_label) != 0))

            bkg_picked_num = np.sum((signal_picked != self.ann_signal_label) != 0)

            if bkg_picked_num > 0:
                bkg_ratios.append(np.sum((self.raw_labels != self.ann_signal_label) != 0) / bkg_picked_num)
            else:
                bkg_ratios.append(-1)

        return np.array(efficiencies), np.array(bkg_ratios)

    def export_ann_info(self,
                        effi_points: list,
                        export: bool = True,
                        detailed: bool = True):
        '''change with increasing thresholds'''

        ann_info_dict = dict()

        ann_effi = []
        ann_puri = []
        ann_bkg_rej = []
        ann_bkg_ra = []
        ann_bkg_effi = []
        ann_thres = []

        ann_start = 0

        ann_info_name = 'ann_info_s_{}_b'.format(self.ann_signal_label)

        while self.ann_signal_label in self.label_list:
            self.label_list.remove(self.ann_signal_label)
        for b in self.label_list:
            ann_info_name = ann_info_name + '_' + str(b)
        # not export all
        # ann_efficiency = self.get_ann_efficiency()
        # ann_purity = self.get_ann_purity()
        # ann_bkg_rate = self.get_ann_bkg_rate()
        # ann_bkg_ratio = self.get_ann_bkg_ratio()
        # ann_bkg_efficiency = self.get_ann_bkg_efficiency()

        # instead
        ann_efficiency, ann_bkg_ratio = self.get_effi_bkg_ratio()
        ann_threshold = self.ann_threshold_lists

        if export:

            for effi in effi_points:

                if ann_start >= len(ann_efficiency):
                    ann_effi.append(ann_efficiency[-1])
                    # ann_puri.append(ann_purity[-1])
                    # ann_bkg_rej.append(ann_bkg_rate[-1])
                    # ann_bkg_effi.append(ann_bkg_efficiency[-1])
                    ann_bkg_ra.append(ann_bkg_ratio[-1])
                    ann_thres.append(ann_threshold[-1])

                for i, _ in enumerate(ann_efficiency[ann_start:]):

                    if _ <= effi:
                        ann_effi.append(_)
                        # ann_puri.append(ann_purity[ann_start:][i])
                        # ann_bkg_rej.append(ann_bkg_rate[ann_start:][i])
                        # ann_bkg_effi.append(ann_bkg_efficiency[ann_start:][i])
                        ann_bkg_ra.append(ann_bkg_ratio[ann_start:][i])
                        ann_thres.append(ann_threshold[ann_start:][i])
                        ann_start = ann_start + i + 1

                        break

            ann_info_dict['ann_thres'] = np.around(np.array(ann_thres), decimals=3)[::-1]
            ann_info_dict['ann_effi'] = np.around(np.array(ann_effi), decimals=3)[::-1]
            # ann_info_dict['ann_puri'] = np.around(np.array(ann_puri), decimals=3)[::-1]
            # ann_info_dict['ann_bkg_rej'] = np.around(np.array(ann_bkg_rej), decimals=3)[::-1]
            ann_info_dict['ann_bkg_ra'] = np.around(np.array(ann_bkg_ra), decimals=3)[::-1]
            # ann_info_dict['ann_bkg_effi'] = np.around(1000 * np.array(ann_bkg_effi), decimals=3)[::-1]

            self.improvement = pd.DataFrame(ann_info_dict, index=np.array(effi_points)[::-1])
            self.improvement.to_csv(os.path.join(self.save_dir, ann_info_name + '.csv'), index=True)

        if detailed:
            ann_info_detailed = dict()
            ann_info_detailed['ann_thres'] = self.ann_threshold_lists
            ann_info_detailed['ann_effi'] = ann_efficiency
            # ann_info_detailed['ann_puri'] = ann_purity
            # ann_info_detailed['ann_bkg_rej'] = ann_bkg_rate
            ann_info_detailed['ann_bkg_ra'] = ann_bkg_ratio
            # ann_info_detailed['ann_bkg_effi'] = ann_bkg_efficiency
            ann_info_detailed = pd.DataFrame(ann_info_detailed)
            ann_info_detailed.to_csv(os.path.join(self.save_dir, ann_info_name + '_detailed.csv'), index=False)

        auc = self.calculate_roc(tpr=ann_efficiency,
                                 fpr=1 / ann_bkg_ratio)

        auc = pd.DataFrame({'auc': [auc]})

        auc.to_csv(os.path.join(self.save_dir, ann_info_name + '_auc.csv'), index=False)

    def calculate_roc(self, tpr, fpr):

        auc = 0
        n = len(tpr)
        for i in range(n - 1):
            auc += (tpr[i] + tpr[i + 1]) * (fpr[i] - fpr[i + 1]) / 2
        return auc

    def plot_confusion_matrix(self):

        candidate_dict = {
            2: ['e', 'pion-'],
            3: ['muon-', 'e', 'pion-', ],
            4: ['muon-', 'e', 'pion-', 'noise'],
        }

        y_true = self.raw_labels
        y_pred = np.argmax(self.ann_scores, axis=1)

        cm = confusion_matrix(y_true=y_true,
                              y_pred=y_pred,
                              normalize='true')
        cm = np.around(cm * 100, decimals=2)
        print(cm)
        # Plot the confusion matrix
        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=candidate_dict.get(self.n_classes),
                    yticklabels=candidate_dict.get(self.n_classes))
        plt.xlabel('Predicted', fontsize=15)
        plt.ylabel('True', fontsize=15)
        plt.title('Confusion Matrix')
        plt.tick_params(axis='both', labelsize=15)
        plt.savefig(os.path.join(self.save_dir, 'cm.png'))
        plt.show()
        plt.close(fig)


class EVAL_ON_CKV(ANN_ANA):

    def __init__(self, ann_scores_path, raw_labels_path, save_dir, ann_threshold_lists, ann_signal_label, n_classes,
                 label_dict,
                 **kwargs):
        super().__init__(ann_scores_path, raw_labels_path, save_dir, ann_threshold_lists, ann_signal_label, n_classes,
                         **kwargs)

        self.ann_scores_2 = read_ann_score(kwargs.get('ann_scores_2_path'), n_classes,
                                           rt_df=False) if ann_scores_path != None else kwargs.get('ann_scores_2')

        self.raw_labels_2 = np.load(kwargs.get('raw_labels_2_path')) if kwargs.get(
            'raw_labels_2_path') != None else kwargs.get('raw_labels_2')

        self.ann_scores = np.concatenate([self.ann_scores[:20000], self.ann_scores_2[:20000]], axis=0)
        self.raw_labels = np.concatenate([self.raw_labels[:20000], self.raw_labels_2[:20000]], axis=0)

        self.label_dict = label_dict

    def plot_bkg_effi_compare(self, y_ll=0, y_ul=1, x_ll=0.9, x_ul=1, x_log=False):
        label_size = 18

        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.text(0.1, 0.89, 'Electron @5GeV\nPion @5GeV', fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.plot(self.get_ann_efficiency(),
                 self.get_ann_bkg_efficiency(), '^-', markersize=6,
                 color='red', label='ANN')

        # make the plot readable

        plt.tick_params(labelsize=label_size, direction='in', length=5)

        plt.ylim(y_ll, y_ul)

        plt.xlim(x_ll, x_ul)
        if x_log:
            plt.xscale('log')
        else:
            plt.xticks(np.linspace(x_ll, x_ul, round(11 - 100 * (1 - x_ul))), fontsize=label_size)

            plt.minorticks_on()
            plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
            plt.xticks(np.linspace(x_ll, x_ul, round(500 * (x_ul - 0.9) + 1)), minor=True)

        # plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=14)

        # plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

        plt.xlabel('{} efficiency '.format(self.label_dict.get(self.ann_signal_label)) + r'$(N_{S}^{sel.}/N_{S})$',
                   fontsize=label_size - 2)
        plt.ylabel('Bkg. efficiency ' + r'$(N_{B}^{sel.}/N_{B}$)', fontsize=label_size - 2)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(0.1, 0.8), bbox_transform=ax.transAxes, loc='upper left', fontsize=label_size - 2)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        fig_name = '{}_bkg_effi'.format(self.label_dict.get(self.ann_signal_label))

        plt.savefig(os.path.join(self.save_dir, fig_name + '.png'))
        plt.show()
        plt.close(fig)

    def plot_purity_compare(self, y_ll=0, y_ul=1, x_ul=1, x_ll=0.9):
        label_size = 18
        text_dict = {
            'mc': 'MC test set\nMC training approach',
            'tb': 'Data test set\nData training approach'
        }

        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.text(0.1, 0.89, 'Electron @5GeV\nPion @5GeV', fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.plot(self.get_ann_efficiency(),
                 self.get_ann_bkg_rate(), '^-', markersize=6,
                 color='red', label='ANN')
        # make the plot readable

        plt.tick_params(labelsize=label_size, direction='in', length=5)
        plt.ylim(y_ll, y_ul + 0.3 * (y_ul - y_ll))
        plt.xlim(x_ll, x_ul)
        plt.xticks(np.linspace(x_ll, x_ul, 11), fontsize=label_size)
        plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=label_size)

        plt.minorticks_on()
        plt.tick_params(labelsize=14, which='minor', direction='in', length=3)
        plt.xticks(np.linspace(x_ll, x_ul, 51), minor=True)
        plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

        plt.xlabel('{} Efficiency '.format(self.label_dict.get(self.ann_signal_label)) + r'$(N_{S}^{sel.}/N_{S})$',
                   fontsize=label_size)
        plt.ylabel('{} purity '.format(
            self.label_dict.get(self.ann_signal_label)) + r'$({N_{S}^{sel.}}/({N_{B}^{sel.}+N_{S}^{sel.}}))$',
                   fontsize=label_size)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.legend(bbox_to_anchor=(0.9, 0.98), bbox_transform=ax.transAxes, loc='upper right', fontsize=label_size - 2)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir,
                                 '{}_purity_effi_compare.png'.format(self.label_dict.get(self.ann_signal_label))))
        plt.show()

        plt.close(fig)


def print_ann_info(file_format, row, column):
    result_dict = dict()
    for file_path in glob.glob(file_format):
        ann_info = pd.read_csv(file_path)
        print(ann_info)
        result_dict[list(file_path.split('/'))[-3]] = ann_info.loc[row, column]  # mean of auc
    result_dict = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    for key, value in result_dict:
        print(key, value)


def geometry_heatmap(file_format: str, b_xy_list: list, b_z_list: list, row: int, column: str, save_dir: str,
                     effi: float, signal: str):
    font_size = 18
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'geoemtry_row_{}.png'.format(row))
    result_dict = dict()

    x_labels = [str(40 * b_xy) + r'$\times$' + str(40 * b_xy) for b_xy in b_xy_list]

    y_labels = [40 // b_z for b_z in b_z_list]

    for b_xy in b_xy_list:
        _ = []
        for b_z in b_z_list:
            file_path = file_format.format(b_xy, b_z)
            ann_info = pd.read_csv(file_path)
            _.append(ann_info.loc[row, column])

        result_dict[b_xy] = _

    df = pd.DataFrame(result_dict,
                      index=b_z_list,
                      columns=b_xy_list,
                      )
    plt.subplots(figsize=(8, 8), facecolor='w')
    ax = plt.gca()
    plt.text(0, 1.15, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )

    plt.text(0, 1.1, 'Bkg. rejection vs different granularity', fontsize=14, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )
    plt.text(0, 1.05, '{} efficiency @{}'.format(signal, effi), fontsize=14, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )

    fig = sns.heatmap(df,
                      vmax=np.amax(df.values),
                      vmin=np.amin(df.values),
                      annot=True,
                      square=True,
                      cmap=sns.diverging_palette(50, 500, n=500),
                      fmt='.1f',
                      annot_kws={"fontsize": font_size, },
                      xticklabels=x_labels,
                      yticklabels=y_labels,
                      )
    fig.get_figure().savefig(save_path, bbox_inches='tight', transparent=True)
    fig.set_xlabel(r'Cell size [$\mathrm{mm^2}$]', fontsize=font_size)
    fig.set_ylabel('Layer number', fontsize=font_size)
    # fig.set_title(title, fontsize=font_size)
    plt.xticks(rotation=45)
    plt.tick_params(axis='both', labelsize=font_size)
    cbar = fig.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)

    plt.show()


def plot_ann_bar(ann_acore_path, n_classes, save_dir, fig_name):
    ann_scores = read_ann_score(
        file_pid_path=ann_acore_path,
        n_classes=n_classes,
        rt_df=True,
    )

    ann_scores = ann_scores.values

    max_index = np.argmax(ann_scores, axis=1)
    counts = collections.Counter(max_index)

    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    xticks_dict = {
        2: ['Electron', 'Pion'],
        3: ['Muon', 'Electron', 'Pion'],
        4: ['Muon', 'Electron', 'Pion', 'Noise'],
    }

    counts = dict(sorted(counts.items()))
    print(counts)

    bar = plt.bar(list(counts.keys()), list(counts.values()), align='center',
                  tick_label=[xticks_dict.get(n_classes)[i] for i in list(counts.keys())], width=0.5)
    plt.bar_label(bar, label_type='edge')
    plt.ylabel('#')

    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold', horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )

    plt.ylim(top=1.2 * max(list(counts.values())))

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, fig_name))
    plt.show()
    plt.close(fig)

    data_df = dict()
    for col, res_ in zip([xticks_dict.get(n_classes)[i] for i in list(counts.keys())], list(counts.values())):
        data_df[col] = [res_]
    df = pd.DataFrame(data_df)
    df.to_csv(os.path.join(save_dir, 'ann.csv'))


def merge_ana_results(file_list: list, column: str, save_path: str):
    print(len(file_list), file_list)
    result = {}

    _ = []

    for path in file_list:

        path_ = os.path.join(path, 'ANA/ann_info_s_1_b_0.csv')
        if os.path.exists(path_):
            df = pd.read_csv(path_, usecols=[column])
            _.append(df.values)

    _ = np.hstack(_)

    avg = np.mean(_, axis=1)
    std = np.std(_, axis=1) / math.sqrt(len(file_list))

    result['bkg_rej_avg'] = avg
    result['bkg_rej_std'] = std

    result = pd.DataFrame(result)

    result.to_csv(save_path)
    print(result)

    _ = []

    result = {}

    for path in file_list:

        path_ = os.path.join(path, 'ANA/roc/auroc.npy')
        if os.path.exists(path_):
            auc = np.load(path_)

            _.append(auc[1])

    avg = np.mean(_, keepdims=True)
    std = np.std(_, keepdims=True) / math.sqrt(len(file_list))

    result['auc_avg'] = avg
    result['auc_std'] = std

    result = pd.DataFrame(result)
    print(result, '\n')


def main_eval_on_FD():
    fd_e_effi = []
    fd_e_puri = []
    ann_e_effi = []
    ann_e_puri = []

    fd_pi_effi = []
    fd_pi_puri = []
    ann_pi_effi = []
    ann_pi_puri = []

    file_root_dir = '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/xiaxin_pid/Digi_MC_calo_{}-_{}GeV'

    for ep in [5, 20, 40, 60, 80]:
        e_dir = file_root_dir.format('e', ep)
        pi_dir = file_root_dir.format('pi', ep)

        fd_e_file = pd.read_csv(os.path.join(e_dir, 'fd.csv'))
        ann_e_file = pd.read_csv(os.path.join(e_dir, 'ann.csv'))

        fd_pi_file = pd.read_csv(os.path.join(pi_dir, 'fd.csv'))
        ann_pi_file = pd.read_csv(os.path.join(pi_dir, 'ann.csv'))

        fd_pi_effi.append(fd_pi_file.loc[0, 'Pion'] / 20000)
        fd_pi_puri.append(fd_pi_file.loc[0, 'Pion'] / (fd_pi_file.loc[0, 'Pion'] + fd_e_file.loc[0, 'Pion']))

        ann_pi_effi.append(ann_pi_file.loc[0, 'Pion'] / 20000)
        ann_pi_puri.append(ann_pi_file.loc[0, 'Pion'] / (ann_pi_file.loc[0, 'Pion'] + ann_e_file.loc[0, 'Pion']))

        fd_e_effi.append(fd_e_file.loc[0, 'Electron'] / 20000)
        fd_e_puri.append(fd_e_file.loc[0, 'Electron'] / (fd_e_file.loc[0, 'Electron'] + fd_pi_file.loc[0, 'Electron']))

        ann_e_effi.append(ann_e_file.loc[0, 'Electron'] / 20000)
        ann_e_puri.append(
            ann_e_file.loc[0, 'Electron'] / (ann_e_file.loc[0, 'Electron'] + ann_pi_file.loc[0, 'Electron']))

    species = [5, 20, 40, 60, 80]

    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold', horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )
    plt.plot(species, fd_pi_effi, '--*', markersize=10, label='FD', color='black')
    plt.plot(species, ann_pi_effi, '--*', markersize=10, label='ANN', color='red')

    plt.xlabel('Energy [MeV]', fontsize=14)
    plt.ylabel('Pion Efficiency', fontsize=14)
    plt.xlim([0, 85])
    plt.xticks(species, fontsize=14)
    plt.ylim([0.995 * min(fd_pi_effi + ann_pi_effi), 1.03 * max(fd_pi_effi + ann_pi_effi)])
    plt.yticks(np.linspace(0.9, 1, 11), fontsize=11)
    plt.legend(loc='lower right')

    plt.show()

    plt.close()

    print((np.array(ann_pi_effi) - np.array(fd_pi_effi)) * 100)

    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold', horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )
    plt.plot(species, fd_pi_puri, '--*', markersize=10, label='FD', color='black')
    plt.plot(species, ann_pi_puri, '--*', markersize=10, label='ANN', color='red')

    plt.xlabel('Energy [MeV]', fontsize=14)
    plt.ylabel('Pion Purity', fontsize=14)
    plt.xlim([0, 85])
    plt.xticks(species, fontsize=14)
    plt.ylim([0.995 * min(fd_pi_puri + ann_pi_puri), 1.005 * max(fd_pi_puri + ann_pi_puri)])
    plt.yticks(np.linspace(0.98, 1, 3), fontsize=12)
    plt.legend(loc='lower right')

    plt.show()

    plt.close()

    print((np.array(ann_pi_puri) - np.array(fd_pi_puri)) * 100)


def get_auc(file_path):
    df = pd.read_csv(file_path)

    tpr = df['ann_effi'].values
    fpr = 1 / df['ann_bkg_ra'].values

    auc = calculate_auc(tpr=tpr, fpr=fpr)

    print(auc)


def plot_bkg_ratio_compare(labels: list, files: list, signal: int, source: str, save_dir: str, y_ll=0, y_ul=1,
                           x_ll=0.95, x_ul=1, **kwargs):
    label_size = 18
    text_dict = {
        'mc': 'MC test set\nMC training approach',
        'tb': 'Data test set\nData training approach'
    }
    signal_dict = {
        0: 'Electron',
        1: 'Pion',

    }

    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )

    # plt.text(0.1, 0.89, text_dict.get(source), fontsize=label_size, fontstyle='normal',
    #          horizontalalignment='left',
    #          verticalalignment='top', transform=ax.transAxes, )
    y_lim_list = []
    # plt.plot(self.get_ann_efficiency(),
    #          self.get_ann_bkg_ratio(), '^-', markersize=5,
    #          color='red', label='ANN')

    for file, label in zip(files, labels):
        df_ = pd.read_csv(file)

        effi = df_['ann_effi'].values
        bkg_r = df_['ann_bkg_ra'].values

        y_lim_list.append(bkg_r[effi >= x_ll])
        plt.plot(effi,
                 bkg_r, '-', markersize=5, label=label, linewidth=3
                 )

        print(label,round(calculate_auc(tpr=effi, fpr=1 / bkg_r), 5))

    # make the plot readable

    plt.tick_params(labelsize=label_size, direction='in', length=5)

    # plt.ylim(y_ll, y_ul)
    plt.xlim(x_ll, x_ul)
    plt.xticks(np.linspace(x_ll, x_ul, 6), fontsize=label_size)
    # plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=14)

    plt.minorticks_on()
    plt.tick_params(labelsize=14, which='minor', direction='in', length=3)
    plt.xticks(np.linspace(x_ll, x_ul, 51), minor=True)
    # plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

    plt.xlabel('{} efficiency '.format(signal_dict.get(signal)) + r'$(N_{S}^{sel.}/N_{S})$', fontsize=label_size)
    plt.ylabel('Bkg. rejection ' + r'$(N_{B}/N_{B}^{sel.}$)', fontsize=label_size)
    plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(0.05, 0.02), bbox_transform=ax.transAxes, loc='lower left', fontsize=label_size - 2)
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        os.path.join(save_dir, kwargs.get('fig_name', 'bkg_ratio_effi_{}_{}_comparae'.format(signal, source)) + '.png'))
    plt.show()
    plt.close(fig)

def calculate_acc(file_path):

    df=pd.read_csv(file_path)

    score=df[['ANN_e', 'ANN_pi']].values

    labels=df['particle_label'].values


    acc= np.sum((np.argmax(score, axis=1) == labels)!=0)/len(labels)
    return acc


if __name__ == '__main__':
    # file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res18_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_4_l_gamma_0.1_step_50_st_True_b_*_v1/ANA/ann_info.csv'
    #
    # print_ann_info(file_format=file_format,
    #                row=0,
    #                column='ann_bkg_ra')

    #

    # file_path_format = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_{}_{}_v1'
    # for b_xy in [1, 2, 3,]:
    #     for b_z in [1, 2, 4,]:
    # # for b_xy in [1]:
    # #     for b_z in [1]:
    #         file_dir = file_path_format.format(b_xy, b_z)
    #         for label in range(3):
    #             ann_ana = ANN_ANA(
    #                 ann_scores_path=os.path.join(file_dir, 'ANA/PIDTags/TV/imgs_ANN.root'),
    #                 raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_{}_{}/TV/labels.npy'.format(
    #                     b_xy, b_z),
    #                 save_dir=os.path.join(file_dir, 'ANA'),
    #                 ann_threshold_lists=np.linspace(0, 0.99999, 10000),
    #                 ann_signal_label=label,
    #                 n_classes=4
    #
    #             )
    #             ann_ana.filter_label(label_list=[0, 1, 2])
    #             ann_ana.export_ann_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1], detailed=True)
    #
    #             _=[0, 1, 2]
    #             _.remove(label)
    #
    #             for b in _:
    #                 ann_ana = ANN_ANA(
    #                     ann_scores_path=os.path.join(file_dir, 'ANA/PIDTags/TV/imgs_ANN.root'),
    #                     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_{}_{}/TV/labels.npy'.format(
    #                         b_xy, b_z),
    #                     save_dir=os.path.join(file_dir, 'ANA'),
    #                     ann_threshold_lists=np.linspace(0, 0.99999, 10000),
    #                     ann_signal_label=label,
    #                     n_classes=4
    #
    #                 )
    #                 ann_ana.filter_label(label_list=[label, b])
    #                 ann_ana.export_ann_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1], detailed=True)
    #
    # for row, effi in enumerate([0.90, 0.93, 0.95, 0.97, 0.99]):
    #     for i, signal in enumerate(['Muon', 'Electron', 'Pion']):
    #         label_list=[0,1,2]
    #         label_list.remove(i)
    #         geometry_heatmap(
    #             file_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_{}_{}_v1' + '/ANA/ann_info_s_{}_b_{}_{}.csv'.format(
    #                 i, label_list[0], label_list[1] ),
    #             b_xy_list=[1, 2, 3, ],
    #             b_z_list=[1, 2, 4][::-1],
    #             row=row,
    #             column='ann_bkg_ra',
    #             save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/0901_{}_granularity_v3'.format(
    #                 signal.lower()),
    #             effi=effi,
    #             signal=signal
    #         )

    # eval=EVAL_ON_CKV(
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_pi_Run123/0615/imgs_ANN.root',
    #     ann_scores_2_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_e_Run133_cut_50/0615/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_pi_Run123/labels.npy',
    #     raw_labels_2_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_e_Run133_cut_50/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/eval_on_ckv_5GeV',
    #     ann_threshold_lists=np.linspace(0, 1, 100000),
    #     ann_signal_label=1,
    #     n_classes=4,
    #     label_dict={0: 'Muon',
    #                 1: 'Electron',
    #                 2: 'Pion',
    #                 3: 'Noise',},
    #
    #
    # )
    # eval.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # eval.plot_bkg_effi_compare(x_ul=1, y_ll=0.00001, x_ll=0, y_ul=1, x_log=False)
    #
    # eval = EVAL_ON_CKV(
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_pi_Run123/0615/imgs_ANN.root',
    #     ann_scores_2_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_e_Run133_cut_50/0615/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_pi_Run123/labels.npy',
    #     raw_labels_2_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_e_Run133_cut_50/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/eval_on_ckv_5GeV',
    #     ann_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=2,
    #     n_classes=4,
    #     label_dict={0: 'Muon',
    #                 1: 'Electron',
    #                 2: 'Pion',
    #                 3: 'Noise', },
    #
    # )
    # eval.plot_purity_compare(x_ul=0.92, x_ll=0.82, y_ll=0.9, y_ul=1)
    # eval.plot_bkg_effi_compare(x_ul=1, y_ll=0.00001, x_ll=0, y_ul=1, x_log=False)

    # eval = EVAL_ON_CKV(
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_pi_Run123_cut_40/0615/imgs_ANN.root',
    #     ann_scores_2_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_e_Run133_cut_50/0615/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_pi_Run123_cut_40/labels.npy',
    #     raw_labels_2_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_e_Run133_cut_50/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/eval_on_ckv_5GeV_40_cut',
    #     ann_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=2,
    #     n_classes=4,
    #     label_dict={0: 'Muon',
    #                 1: 'Electron',
    #                 2: 'Pion',
    #                 3: 'Noise', },
    #
    # )
    # eval.plot_purity_compare(x_ul=1, x_ll=0.9, y_ll=0.9, y_ul=1)
    # eval.plot_bkg_effi_compare(x_ul=1, y_ll=0.00001, x_ll=0, y_ul=1, x_log=False)

    # plot_ann_bar(
    #    ann_acore_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_pi_Run123/0615/imgs_ANN.root',
    #    n_classes=4,
    #    save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig/eval_on_ckv_5GeV',
    #    fig_name='5gev_pi.png'
    # )
    #
    # file_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/xiaxin_pid/Digi_MC_calo_{}-_{}GeV'
    #
    # for ep in [5, 20, 40, 60, 80]:
    #
    #     for particle in ['e', 'pi']:
    #
    #
    #         file_dir=file_root_dir.format(particle, ep)
    #
    #         plot_ann_bar(
    #             ann_acore_path=file_dir+'/0915/imgs_ANN.root',
    #             n_classes=2,
    #             save_dir=file_dir,
    #             fig_name='ANN_classification.png'
    #         )

    # main_eval_on_FD()

    # ann_ana = ANN_ANA(
    #                     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/PIDTags/Test_2/imgs_ANN.root',
    #                     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/mc_0720_e_pi_2/Test/labels.npy',
    #                     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/Fig',
    #                     ann_threshold_lists=np.linspace(0, 0.99999, 10000),
    #                     ann_signal_label=1,
    #                     n_classes=2,
    #                 )
    #
    # ann_ana.plot_confusion_matrix()
    #
    # ann_ana = ANN_ANA(
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_False_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/PIDTags/Test_2/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/mc_0720_e_pi_2/Test/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_False_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/Fig',
    #     ann_threshold_lists=np.linspace(0, 0.99999, 10000),
    #     ann_signal_label=1,
    #     n_classes=2,
    # )
    #
    # ann_ana.plot_confusion_matrix()

    # ana_scores_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/PIDTags/Test_2/imgs_ANN.root'
    # n_classes = 2
    #
    # ann_scores = read_ann_score(
    #     file_pid_path=ana_scores_path,
    #     n_classes=n_classes,
    #     rt_df=False
    # )
    # ann_threshold_lists = np.sort(ann_scores[:, 1])
    # ann_threshold_lists = np.unique(ann_threshold_lists)
    # print(ann_threshold_lists.shape)

    # ann_ana = ANN_ANA(
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_False_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/PIDTags/Test_2/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/mc_0720_e_pi_2/Test/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_False_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/Fig',
    #     ann_threshold_lists=np.linspace(0, 0.99999, 10000),
    #     ann_signal_label=1,
    #     n_classes=2,
    # )
    #
    # ann_ana.plot_confusion_matrix()

    # merge_ana_results(file_list=['/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_0_mc_resnet_avg_epoch_200_lr_0.0001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_100_st_0_fk_3_fs_1_fp_1_v1/ANA/ann_info_s_1_b_0.csv',
    #                              '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_2_mc_resnet_avg_epoch_200_lr_0.0001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_100_st_0_fk_3_fs_1_fp_1_v1/ANA/ann_info_s_1_b_0.csv',
    #                              '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_3_mc_resnet_avg_epoch_200_lr_0.0001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_100_st_0_fk_3_fs_1_fp_1_v1/ANA/ann_info_s_1_b_0.csv',
    #                              '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_4_mc_resnet_avg_epoch_200_lr_0.0001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_100_st_0_fk_3_fs_1_fp_1_v1/ANA/ann_info_s_1_b_0.csv',
    #                              '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_5_mc_resnet_avg_epoch_200_lr_0.0001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_100_st_0_fk_3_fs_1_fp_1_v1/ANA/ann_info_s_1_b_0.csv',
    #
    #                              ],
    #
    #                   column='ann_bkg_ra',
    #                   save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/resnet_st_0.csv')

    # merge_ana_results(file_list= glob.glob(
    #     '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_*_mc_resnet_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1'),
    #
    #                   column='ann_bkg_ra',
    #                   save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/resnet_st_1.csv')
    #
    #
    #
    # merge_ana_results(file_list=glob.glob(
    #     '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_*_mc_dgres_epoch_30_lr_0.01_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_10_fk_3_fs_1_fp_1_v1'),
    #
    #     column='ann_bkg_ra',
    #     save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/dgres_5.csv')
    #
    # merge_ana_results(file_list=glob.glob(
    #     '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_*_mc_dgres_epoch_30_lr_0.001_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_10_fk_3_fs_1_fp_1_v1'),
    #
    #     column='ann_bkg_ra',
    #     save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/dgres_7.csv')
    #
    # merge_ana_results(file_list=glob.glob(
    #     '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119_*_mc_dgcnn_epoch_30_lr_0.1_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_64_k_20_v1'),
    #
    #     column='ann_bkg_ra',
    #     save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/dgcnn_2.csv')
    #
    # merge_ana_results(file_list=glob.glob(
    #     '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119_*_mc_dgcnn_epoch_30_lr_0.1_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_128_k_20_v1'),
    #
    #     column='ann_bkg_ra',
    #     save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/dgcnn.csv')
    #
    # merge_ana_results(file_list=glob.glob(
    #     '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119_*_mc_dgcnn_epoch_30_lr_0.1_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_128_k_10_v1'),
    #
    #     column='ann_bkg_ra',
    #     save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/dgcnn.csv')
    #
    #
    # merge_ana_results(file_list=glob.glob(
    #     '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0122_*_mc_gravnet_epoch_30_lr_0.001_batch_32_optim_SGD_classes_2_nodes_256_k_10_v1'),
    #
    #     column='ann_bkg_ra',
    #     save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/gravnet.csv')

    # dgcnn_2 = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119_4_mc_dgcnn_epoch_30_lr_0.1_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_64_k_20_v1ANA/ann_info_s_1_b_0.csv'
    # res_net = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_29_mc_resnet_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_1_fk_7_fs_2_fp_3_v1/ANA/ann_info_s_1_b_0.csv'
    # dgres='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_26_mc_dgres_epoch_30_lr_0.1_batch_64_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_40_fk_7_fs_2_fp_3_v1/ANA/ann_info_s_1_b_0.csv'
    # gravnet='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0122_34_mc_gravnet_epoch_30_lr_0.001_batch_32_optim_SGD_classes_2_nodes_256_k_10_v1'
    # gravnet_2 = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0122_7_mc_gravnet_epoch_30_lr_0.001_batch_32_optim_SGD_classes_2_nodes_128_k_64_v1/ANA/ann_info_s_1_b_0.csv'
    # dgcnn='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119_2_mc_dgcnn_epoch_30_lr_0.1_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_128_k_10_v1'
    # dgres_2 = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_5_mc_dgres_epoch_30_lr_0.1_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_20_fk_3_fs_1_fp_1_v1/ANA/ann_info_s_1_b_0.csv'

    # dgcnn_3 = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119_2_mc_dgcnn_epoch_30_lr_0.1_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_128_k_10_v1/ANA/ann_info_s_1_b_0.csv'
    # dgres_3 = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_47_mc_dgres_epoch_30_lr_0.01_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_40_fk_3_fs_1_fp_1_v1/ANA/ann_info_s_1_b_0.csv'
    # dgres_4 = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_54_mc_dgres_epoch_30_lr_0.001_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_128_k_10_fk_3_fs_1_fp_1_v1/ANA/ann_info_s_1_b_0.csv'
    # dgres_5 = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_43_mc_dgres_epoch_30_lr_0.01_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_10_fk_3_fs_1_fp_1_v1/ANA/ann_info_s_1_b_0.csv'
    # dgres_6 = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_67_mc_dgres_epoch_30_lr_0.01_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_10_fk_3_fs_1_fp_1_v1'
    # dgres_7 = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_83_mc_dgres_epoch_30_lr_0.001_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_10_fk_3_fs_1_fp_1_v1'
    # dgres_8 = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_107_mc_dgres_epoch_30_lr_0.01_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_10_fk_3_fs_1_fp_1_v1'
    #
    # ana_detail = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0118_10_mc_lenet_epoch_200_lr_0.01_batch_64_optim_SGD_classes_2_l_gamma_0.1_step_10_v1/ANA/ann_info_s_1_b_0_detailed.csv'

    # get_auc(file_path=ana_detail)
    # v1: 0119 81 0121 107 0122 39
    # v2: 0119 77, 0121 107, 0122 100
    # v3: 0119 81, 0121 107, 0122 100
    # v4: 0119 99, 0121 107, 0122 100
    # v5: 0119 81, 0121 67, 0122 39
    # current : v5
    plot_bkg_ratio_compare(labels=['BDT', 'LeNet', 'AlexNet', 'ResNet' ],
                           files=[
                               '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_draft_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/bdt_detailed_s_1_b_0.csv',
                               '/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/lenet_epoch_200_lr_1e-05_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_v1.csv',
                               '/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/alexnet_epoch_30_lr_0.001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_v1.csv',
                               # '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_230_mc_resnet_avg_epoch_200_lr_1e-05_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_st_1_fk_3_fs_1_fp_1_v1/ANA/ann_info_s_1_b_0_detailed.csv',
                               '/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/used_resnet_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_0.1_step_10_st_1_fk_3_fs_1_fp_1_v1.csv'
                           ],
                           source='mc',
                           signal=1,
                           save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig',
                           fig_name='point_cloud_scenario_compare_bdt_imgbased',
                           x_ll=0.95
                           )

    plot_bkg_ratio_compare(labels=['DGCNN', 'Gravnet', 'LeNet', 'AlexNet', 'BDT', 'DGRes', 'ResNet'],
                           files=[
                               '/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/dgcnn_epoch_30_lr_0.1_batch_128_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_128_k_20_v1.csv',
                               '/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/gravnet_epoch_30_lr_0.001_batch_64_optim_SGD_classes_2_nodes_256_k_10_v1.csv',
                               '/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/lenet_epoch_200_lr_1e-05_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_v1.csv',
                               '/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/alexnet_epoch_30_lr_0.001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_v1.csv',

                               '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_draft_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/bdt_detailed_s_1_b_0.csv',
                               '/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/dgres_epoch_30_lr_0.01_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_10_fk_3_fs_1_fp_1_v1.csv',
                               '/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/result/used_resnet_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_0.1_step_10_st_1_fk_3_fs_1_fp_1_v1.csv',
                           ],
                           source='mc',
                           signal=1,
                           save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig',
                           fig_name='point_cloud_scenario_compare_v2',
                           x_ll=0.95
                           )

    # plot_bkg_ratio_compare(labels=['DGCNN', 'Gravnet', 'LeNet', 'AlexNet', 'GoogleNet', 'BDT', 'DGRes', 'ResNet'],
    #                        files=[
    #                            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119_80_mc_dgcnn_epoch_30_lr_0.1_batch_128_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_128_k_20_v1/ANA/ann_info_s_1_b_0_detailed.csv',
    #                            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0122_39_mc_gravnet_epoch_30_lr_0.001_batch_64_optim_SGD_classes_2_nodes_256_k_10_v1/ANA/ann_info_s_1_b_detailed.csv',
    #                            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0118_15_mc_lenet_epoch_200_lr_1e-05_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_v1/ANA/ann_info_s_1_b_0_detailed.csv',
    #                            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0116_22_mc_alexnet_epoch_30_lr_0.001_batch_32_optim_SGD_classes_2_l_gamma_1.0_step_10_v1/ANA/ann_info_s_1_b_0_detailed.csv',
    #                            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0117_40_mc_googlenet_epoch_30_lr_0.01_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_v1/ANA/ann_info_s_1_b_0_detailed.csv',
    #                            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_draft_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/bdt_detailed_s_1_b_0.csv',
    #                            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_67_mc_dgres_epoch_30_lr_0.01_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_10_fk_3_fs_1_fp_1_v1/ANA/ann_info_s_1_b_detailed.csv',
    #                            '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_draft_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/ann_detailed_s_1_b_0.csv',
    #                        ],
    #                        source='mc',
    #                        signal=1,
    #                        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/Fig',
    #                        fig_name='point_cloud_scenario_compare_v2',
    #                        x_ll=0.95
    #                        )

    labels = ['DGCNN', 'Gravnet',  'DGRes',  ]
    files = [
                '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0119_80_mc_dgcnn_epoch_30_lr_0.1_batch_128_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_128_k_20_v1/ANA/imgs_ANN.csv',
                '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0122_39_mc_gravnet_epoch_30_lr_0.001_batch_64_optim_SGD_classes_2_nodes_256_k_10_v1/ANA/imgs_ANN.csv',
                # '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0118_15_mc_lenet_epoch_200_lr_1e-05_batch_256_optim_SGD_classes_2_l_gamma_1.0_step_10_v1/ANA/ann_info_s_1_b_0_detailed.csv',

                '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0121_67_mc_dgres_epoch_30_lr_0.01_batch_32_optim_SGD_classes_2_l_gamma_0.1_step_10_nodes_256_k_10_fk_3_fs_1_fp_1_v1/ANA/imgs_ANN.csv',

            ]

    for label, file in zip(labels, files):

        print(label, round(100*calculate_acc(file_path=file), 3))
    pass
