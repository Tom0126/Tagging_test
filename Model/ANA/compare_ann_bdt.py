#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/23 14:43
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : compare_ann_bdt.py
# @Software: PyCharm

from e_sigma_reconstruct import read_ann_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
import copy

class Compare():

    def __init__(self,bdt_eval_path, ann_scores_path, raw_labels_path, save_dir, ann_threshold_lists, bdt_threshold_lists,
                 ann_signal_label, source, n_classes, label_dict, label_list, **kwargs):

        self.source=source
        self.save_dir=save_dir
        os.makedirs(self.save_dir,exist_ok=True)
        self.bdt_eval = pd.read_csv(bdt_eval_path)
        self.ann_scores= read_ann_score(ann_scores_path, n_classes,rt_df=False) if ann_scores_path !=None else kwargs.get('ann_scores')
        self.raw_labels = np.load(raw_labels_path) if raw_labels_path != None else kwargs.get(
            'raw_labels')
        self.ann_threshold_lists=ann_threshold_lists
        self.bdt_threshold_lists = bdt_threshold_lists
        self.ann_signal_label=ann_signal_label

        self.label_dict = label_dict
        self.label_list = label_list

    def filter_label(self):

        cut_ann=self.raw_labels==self.label_list[0]
        cut_bdt=self.bdt_eval['raw_labels']==self.label_list[0]

        for label in self.label_list:
            cut_ann=np.logical_or(cut_ann, self.raw_labels==label)
            cut_bdt = np.logical_or(cut_bdt, self.bdt_eval['raw_labels'] == label)

        self.ann_scores=self.ann_scores[cut_ann]
        self.raw_labels=self.raw_labels[cut_ann]
        self.bdt_eval=self.bdt_eval[cut_bdt]

    def get_ann_ns_nb(self):

        ns_ann=[]
        nb_ann=[]

        predictions=self.ann_scores[:, self.ann_signal_label]
        labels=self.raw_labels

        for i, cut in enumerate(self.ann_threshold_lists):
            ns_ann.append(np.sum((predictions[labels==self.ann_signal_label]>=cut)!=0))
            nb_ann.append(np.sum((predictions[labels!=self.ann_signal_label]>=cut)!=0))

        self.ns_ann=np.array(ns_ann)
        self.nb_ann=np.array(nb_ann)

    def get_ann_purity(self):
        return self.ns_ann / (self.ns_ann + self.nb_ann)

    def get_ann_efficiency(self):
        return self.ns_ann / np.sum((self.raw_labels == self.ann_signal_label) != 0)

    def get_ann_bkg_rate(self):
        return 1 - self.nb_ann / np.sum((self.raw_labels != self.ann_signal_label) != 0)

    def get_ann_bkg_ratio(self):
        return np.where(self.nb_ann == 0, -1, np.sum((self.raw_labels != self.ann_signal_label) != 0) / self.nb_ann)

    def get_ann_bkg_efficiency(self):
        return self.nb_ann /np.sum((self.raw_labels != self.ann_signal_label) != 0)

    def get_bdt_ns_nb(self):

        ns_bdt=[]
        nb_bdt=[]

        predictions=self.bdt_eval['predictions'].values
        labels=self.bdt_eval['labels'].values
        for i, cut in enumerate(self.bdt_threshold_lists):
            ns_bdt.append(np.sum((predictions[labels==1]>=cut)!=0))
            nb_bdt.append( np.sum((predictions[labels==0]>=cut)!=0))

        self.ns_bdt=np.array(ns_bdt)
        self.nb_bdt=np.array(nb_bdt)

    def get_bdt_purity(self):
        return self.ns_bdt/ (self.ns_bdt+self.nb_bdt)

    def get_bdt_efficiency(self):
        return self.ns_bdt / np.sum((self.bdt_eval['labels'].values==1)!=0)

    def get_bdt_bkg_rate(self):
        return 1-self.nb_bdt/ np.sum((self.bdt_eval['labels'].values==0)!=0)

    def get_bdt_bkg_ratio(self):
        return np.where(self.nb_bdt ==0, -1, np.sum((self.bdt_eval['labels'].values==0)!=0) / self.nb_bdt)

    def get_bdt_bkg_efficiency(self):
        return self.nb_bdt/ np.sum((self.bdt_eval['labels'].values==0)!=0)

    def export_info(self):

        ann_name = 'ann_detailed_s_{}_b'.format(self.ann_signal_label)
        bdt_name = 'bdt_detailed_s_{}_b'.format(self.ann_signal_label)

        label_list = copy.deepcopy(self.label_list)
        while self.ann_signal_label in label_list:
            label_list.remove(self.ann_signal_label)
        for b in label_list:
            ann_name = ann_name + '_' + str(b)
            bdt_name = bdt_name + '_' + str(b)

        self.df_ann=pd.DataFrame({
            'threshold':self.ann_threshold_lists,
            'ann_effi':self.get_ann_efficiency(),
            # 'ann_bkg_rate':self.get_ann_bkg_rate(),
            'ann_bkg_ra': self.get_ann_bkg_ratio(),
            # 'ann_puri':self.get_ann_purity(),
            # 'ann_bkg_effi': self.get_ann_bkg_efficiency(),
        })

        self.df_ann.to_csv(os.path.join(self.save_dir, ann_name+'.csv'), index=False)

        self.df_bdt = pd.DataFrame({
            'threshold': self.bdt_threshold_lists,
            'ann_effi': self.get_bdt_efficiency(),
            # 'bdt_bkg_rate': self.get_bdt_bkg_rate(),
            'ann_bkg_ra': self.get_bdt_bkg_ratio(),
            # 'bdt_puri': self.get_bdt_purity(),
            # 'bdt_bkg_effi': self.get_bdt_bkg_efficiency(),
        })

        self.df_bdt.to_csv(os.path.join(self.save_dir, bdt_name+'.csv'), index=False)

    def export_improvement_info(self, effi_points:list):
        '''change with increasing thresholds'''

        imp_dict=dict()

        ann_threshold=[]
        bdt_threshold = []

        ann_effi=[]
        bdt_effi=[]

        ann_puri=[]
        bdt_puri=[]

        ann_bkg_rej=[]
        bdt_bkg_rej=[]

        ann_bkg_ra = []
        bdt_bkg_ra= []

        ann_bkg_effi=[]
        bdt_bkg_effi=[]

        ann_start=0
        bdt_start=0

        ann_efficiency = self.get_ann_efficiency()
        ann_purity=self.get_ann_purity()
        ann_bkg_rate=self.get_ann_bkg_rate()
        ann_bkg_ratio = self.get_ann_bkg_ratio()
        ann_bkg_efficiency=self.get_ann_bkg_efficiency()

        bdt_efficiency = self.get_bdt_efficiency()
        bdt_purity = self.get_bdt_purity()
        bdt_bkg_rate = self.get_bdt_bkg_rate()
        bdt_bkg_ratio = self.get_bdt_bkg_ratio()
        bdt_bkg_efficiency = self.get_bdt_bkg_efficiency()

        for effi in effi_points:


            if ann_start >= len(ann_efficiency):

                ann_threshold.append(self.ann_threshold_lists[-1])
                ann_effi.append(ann_efficiency[-1])
                ann_puri.append(ann_purity[-1])
                ann_bkg_rej.append(ann_bkg_rate[-1])
                ann_bkg_effi.append(ann_bkg_efficiency[-1])
                ann_bkg_ra.append(ann_bkg_ratio[-1])



            for i, _ in enumerate(ann_efficiency[ann_start:]):

                if _ <= effi:

                    ann_effi.append(_)
                    ann_threshold.append(self.ann_threshold_lists[ann_start:][i])
                    ann_puri.append(ann_purity[ann_start:][i])
                    ann_bkg_rej.append(ann_bkg_rate[ann_start:][i])
                    ann_bkg_effi.append(ann_bkg_efficiency[ann_start:][i])
                    ann_bkg_ra.append(ann_bkg_ratio[ann_start:][i])
                    ann_start = ann_start + i + 1

                    break


            if bdt_start >= len(bdt_efficiency):
                bdt_threshold.append(self.bdt_threshold_lists[-1])
                bdt_effi.append(bdt_efficiency[-1])
                bdt_puri.append(bdt_purity[-1])
                bdt_bkg_rej.append(bdt_bkg_rate[-1])
                bdt_bkg_effi.append(bdt_bkg_efficiency[-1])
                bdt_bkg_ra.append(bdt_bkg_ratio[-1])

            for i, _ in enumerate(bdt_efficiency[bdt_start:]):
                if _ <= effi:
                    bdt_effi.append(_)
                    bdt_threshold.append(self.bdt_threshold_lists[bdt_start:][i])
                    bdt_puri.append(bdt_purity[bdt_start:][i])
                    bdt_bkg_rej.append(bdt_bkg_rate[bdt_start:][i])
                    bdt_bkg_effi.append(bdt_bkg_efficiency[bdt_start:][i])
                    bdt_bkg_ra.append(bdt_bkg_ratio[bdt_start:][i])
                    bdt_start = bdt_start + i + 1

                    break

        imp_dict['ann_thre']=ann_threshold[::-1]
        imp_dict['bdt_thre'] = bdt_threshold[::-1]

        imp_dict['ann_effi'] = np.around(np.array(ann_effi), decimals=3)[::-1]
        imp_dict['bdt_effi'] = np.around(np.array(bdt_effi), decimals=3)[::-1]

        imp_dict['bdt_puri']=np.around(np.array(bdt_puri),decimals=3)[::-1]
        imp_dict['ann_puri']=np.around(np.array(ann_puri),decimals=3)[::-1]
        imp_dict['puri_imp']=np.around(100*(np.array(ann_puri)-np.array(bdt_puri))/np.array(bdt_puri),decimals=3)[::-1]

        imp_dict['bdt_bkg_rej'] = np.around(np.array(bdt_bkg_rej),decimals=3)[::-1]
        imp_dict['ann_bkg_rej'] = np.around(np.array(ann_bkg_rej),decimals=3)[::-1]
        imp_dict['bkg_rej_imp'] = np.around(100 * (np.array(ann_bkg_rej) - np.array(bdt_bkg_rej)) / np.array(bdt_bkg_rej),decimals=3)[::-1]

        imp_dict['bdt_bkg_ra'] = np.around(np.array(bdt_bkg_ra), decimals=3)[::-1]
        imp_dict['ann_bkg_ra'] = np.around(np.array(ann_bkg_ra), decimals=3)[::-1]
        imp_dict['bkg_ra_imp'] = np.around(
            100 * (np.array(ann_bkg_ra) - np.array(bdt_bkg_ra)) / np.array(bdt_bkg_ra), decimals=3)[::-1]

        imp_dict['bdt_bkg_effi'] = np.around(1000*np.array(bdt_bkg_effi),decimals=3)[::-1]
        imp_dict['ann_bkg_effi'] = np.around(1000*np.array(ann_bkg_effi),decimals=3)[::-1]
        imp_dict['bkg_effi_imp'] = np.around(100 * (np.array(ann_bkg_effi) - np.array(bdt_bkg_effi)) / np.array(bdt_bkg_effi),decimals=3)[::-1]

        imp_name = 'ann_info_s_{}_b'.format(self.ann_signal_label)

        label_list=copy.deepcopy(self.label_list)
        while self.ann_signal_label in label_list:
            label_list.remove(self.ann_signal_label)
        for b in label_list:
            imp_name = imp_name + '_' + str(b)

        self.improvement=pd.DataFrame(imp_dict, index=np.array(effi_points)[::-1])
        self.improvement.to_csv(os.path.join(self.save_dir, imp_name+'.csv'), index=True)



    def plot_purity_compare(self, y_ll=0, y_ul=1,x_ul=1):
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

        plt.text(0.1, 0.89, text_dict.get(self.source), fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.plot(self.get_ann_efficiency(),
                 self.get_ann_purity(), '^-', markersize=6,
                 color='red', label='ANN')

        plt.plot(self.get_bdt_efficiency(),
                 self.get_bdt_purity(), 'o-', markersize=6,
                 color='blue', label='BDT')

        # make the plot readable

        plt.tick_params(labelsize=label_size, direction='in', length=5)
        plt.ylim(y_ll, y_ul+0.3*(y_ul-y_ll))
        plt.xlim(0.9, x_ul)
        plt.xticks(np.linspace(0.9, x_ul, round(11-100*(1-x_ul))), fontsize=label_size)
        plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=label_size)

        plt.minorticks_on()
        plt.tick_params(labelsize=14, which='minor', direction='in', length=3)
        plt.xticks(np.linspace(0.9, x_ul, round(500*(x_ul-0.9)+1)), minor=True)
        plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

        plt.xlabel('{} Efficiency '.format(self.label_dict.get(self.ann_signal_label))+r'$(N_{S}^{sel.}/N_{S})$', fontsize=label_size)
        plt.ylabel('{} purity '.format(self.label_dict.get(self.ann_signal_label)) + r'$({N_{S}^{sel.}}/({N_{B}^{sel.}+N_{S}^{sel.}}))$', fontsize=label_size)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.legend(bbox_to_anchor=(0.9, 0.98), bbox_transform=ax.transAxes, loc='upper right', fontsize=label_size-2)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir, '{}_purity_effi_{}_compare.png'.format(self.label_dict.get(self.ann_signal_label),self.source)))
        plt.close(fig)

    def plot_bkg_rej_compare(self, y_ll=0, y_ul=1, x_ll=0, x_ul=1):

        text_dict = {
            'mc': 'MC test set, MC training approach',
            'tb': 'Data test set, Data training approach'
        }

        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.text(0.1, 0.89, text_dict.get(self.source), fontsize=14, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.plot(self.get_ann_efficiency(),
                 self.get_ann_bkg_rate(), '^-', markersize=6,
                 color='red', label='ANN')

        plt.plot(self.get_bdt_efficiency(),
                 self.get_bdt_bkg_rate(), 'o-', markersize=6,
                 color='blue', label='BDT')

        # make the plot readable

        plt.tick_params(labelsize=14, direction='in', length=5)

        plt.ylim(y_ll, y_ul + 0.2 * (y_ul - y_ll))
        plt.xlim(0.9, x_ul)
        plt.xticks(np.linspace(0.9, x_ul, round(11 - 100 * (1 - x_ul))), fontsize=14)
        plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=14)

        plt.minorticks_on()
        plt.tick_params(labelsize=14,which='minor', direction='in', length=3)
        plt.xticks(np.linspace(0.9, x_ul, round(500 * (x_ul - 0.9) + 1)), minor=True)
        plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)


        plt.xlabel('{} efficiency '.format(self.label_dict.get(self.ann_signal_label))+r'$(N_{S}^{sel.}/N_{S})$', fontsize=14)
        plt.ylabel('Bkg. rejection rate '+r'$(1- N_{B}^{sel.}/N_{B}$)', fontsize=14)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.legend(bbox_to_anchor=(0.9, 0.98), bbox_transform=ax.transAxes, loc='upper right', fontsize=14)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir, 'bkg_rej_effi_{}_compare.png'.format(self.source)))
        plt.close(fig)

    def plot_bkg_ratio_compare(self,y_ll=0, y_ul=1, x_ll=0.9, x_ul=1, **kwargs):
        label_size = 18
        text_dict = {
            'mc': 'MC test set\nMC training approach',
            'tb': 'Data test set\nData training approach'
        }

        if 'ann_file_path' in kwargs.keys():
            ann_file=pd.read_csv(kwargs.get('ann_file_path'))
            ann_effi=ann_file['ann_effi'].values
            ann_bkg_ratio=ann_file['ann_bkg_ra'].values
        else:

            ann_effi=self.get_ann_efficiency()
            ann_bkg_ratio=self.get_ann_bkg_ratio()

        if 'bdt_file_path' in kwargs.keys():
            bdt_file=pd.read_csv(kwargs.get('bdt_file_path'))
            bdt_effi=bdt_file['ann_effi'].values
            bdt_bkg_ratio=bdt_file['ann_bkg_ra'].values
        else:
            bdt_effi=self.get_bdt_efficiency()
            bdt_bkg_ratio=self.get_bdt_bkg_ratio()

        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.text(0.1, 0.89, text_dict.get(self.source), fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.plot(ann_effi[ann_bkg_ratio!=-1],
                 ann_bkg_ratio[ann_bkg_ratio!=-1], '-', markersize=6,
                 linewidth=4,
                 color='red', label='ResNet')

        plt.plot(bdt_effi[bdt_bkg_ratio!=-1],
                 bdt_bkg_ratio[bdt_bkg_ratio!=-1], '-', markersize=6,
                 linewidth=4,
                 color='blue', label='BDT')
        # make the plot readable

        plt.tick_params(labelsize=label_size, direction='in', length=5)

        plt.ylim(y_ll, y_ul)
        plt.xlim(x_ll, x_ul)
        plt.xticks(np.linspace(x_ll, x_ul, 6), fontsize=label_size)
        # plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=14)

        plt.minorticks_on()
        plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
        plt.xticks(np.linspace(x_ll, x_ul, 51), minor=True)
        # plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

        plt.xlabel('{} efficiency '.format(self.label_dict.get(self.ann_signal_label)) + r'$(N_{S}^{sel.}/N_{S})$', fontsize=label_size)
        plt.ylabel('Bkg. rejection ' + r'$(N_{B}/N_{B}^{sel.}$)', fontsize=label_size)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(0.9, 0.98), bbox_transform=ax.transAxes, loc='upper right', fontsize=label_size-2)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        fig_name = '{}_bkg_ratio_effi_{}_compare'.format(self.label_dict.get(self.ann_signal_label),self.source)
        label_list=copy.deepcopy(self.label_list)

        while self.ann_signal_label in label_list:
            label_list.remove(self.ann_signal_label)
        for b in label_list:
            fig_name = fig_name + '_' + str(b)


        plt.savefig(os.path.join(self.save_dir, fig_name+'.png'))
        plt.show()
        plt.close(fig)


    def plot_bkg_effi_compare(self,y_ll=0, y_ul=1, x_ll=0.9, x_ul=1, x_log=False, y_log=False, **kwargs):
        label_size = 18
        text_dict = {
            'mc': 'MC test set\nMC training approach',
            'tb': 'Data test set\nData training approach'
        }

        if 'ann_file_path' in kwargs.keys():
            ann_file = pd.read_csv(kwargs.get('ann_file_path'))
            ann_effi = ann_file['ann_effi'].values
            ann_bkg_effi = ann_file['ann_bkg_effi'].values
        else:

            ann_effi = self.get_ann_efficiency()
            ann_bkg_effi = self.get_ann_bkg_efficiency()

        if 'bdt_file_path' in kwargs.keys():
            bdt_file = pd.read_csv(kwargs.get('bdt_file_path'))
            bdt_effi = bdt_file['bdt_effi'].values
            bdt_bkg_effi = bdt_file['bdt_bkg_effi'].values
        else:
            bdt_effi = self.get_bdt_efficiency()
            bdt_bkg_effi = self.get_bdt_bkg_efficiency()

        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.text(0.1, 0.89, text_dict.get(self.source), fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.plot(ann_effi,
                 ann_bkg_effi, '^-', markersize=6,
                 color='red', label='ANN')


        plt.plot(bdt_effi,
                 bdt_bkg_effi, 'o-', markersize=6,
                 color='blue', label='BDT')
        # make the plot readable

        plt.tick_params(labelsize=label_size, direction='in', length=5)

        plt.ylim(y_ll, y_ul)

        plt.xlim(x_ll, x_ul)
        if x_log:
            plt.xscale('log')
        else:
            plt.xticks(np.linspace(x_ll, x_ul, 11), fontsize=label_size)

            plt.minorticks_on()
            plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
            plt.xticks(np.linspace(x_ll, x_ul, 51), minor=True)

        if y_log:
            plt.yscale('log')
        else:
            plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=label_size)

            plt.minorticks_on()
            plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
            plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)


        plt.xlabel('{} efficiency '.format(self.label_dict.get(self.ann_signal_label)) + r'$(N_{S}^{sel.}/N_{S})$', fontsize=label_size-2)
        plt.ylabel('Bkg. efficiency ' + r'$(N_{B}^{sel.}/N_{B}$)', fontsize=label_size-2)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        plt.legend(bbox_to_anchor=(0.1, 0.8), bbox_transform=ax.transAxes, loc='upper left', fontsize=label_size-2)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        fig_name = '{}_bkg_effi_{}_compare'.format(self.label_dict.get(self.ann_signal_label),self.source)
        label_list=copy.deepcopy(self.label_list)

        while self.ann_signal_label in label_list:
            label_list.remove(self.ann_signal_label)
        for b in label_list:
            fig_name = fig_name + '_' + str(b)


        plt.savefig(os.path.join(self.save_dir, fig_name+'.png'))
        plt.show()
        plt.close(fig)

    def plot_bkg_ratio_imp_factor(self,imp_file_path, x_col, y_col, y_ll=0, y_ul=1, x_ll=0.9, x_ul=1, x_log=False, y_log=False, **kwargs):

        label_size = 18
        text_dict = {
            'mc': 'MC test set\nMC training approach',
            'tb': 'Data test set\nData training approach'
        }

        imp_file=pd.read_csv(imp_file_path, usecols=[x_col, y_col])

        x=imp_file[x_col][::24]
        y=imp_file[y_col][::24]

        if 'y_scale' in kwargs.keys():
            y=y*kwargs.get('y_scale')

        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.text(0.1, 0.89, text_dict.get(self.source), fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.plot(x,
                 y,
                 'o--',
                 markersize=6,
                 linewidth=3,
                 color='red',
                 label='ANN',)

        # make the plot readable

        plt.tick_params(labelsize=label_size, direction='in', length=5)

        plt.ylim(y_ll, y_ul)

        plt.xlim(x_ll, x_ul)
        if x_log:
            plt.xscale('log')
        else:
            plt.xticks(np.linspace(x_ll, x_ul, 5), fontsize=label_size)

            plt.minorticks_on()
            plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
            plt.xticks(np.linspace(x_ll, x_ul, 51), minor=True)

        if y_log:
            plt.yscale('log')
        else:
            plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=label_size)

            plt.minorticks_on()
            plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
            plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

        plt.xlabel('{} efficiency '.format(self.label_dict.get(self.ann_signal_label)) + r'$(N_{S}^{sel.}/N_{S})$',
                   fontsize=label_size - 2)
        plt.ylabel('Improvement factor', fontsize=label_size - 2)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        plt.legend(bbox_to_anchor=(0.1, 0.8), bbox_transform=ax.transAxes, loc='upper left', fontsize=label_size - 2)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        fig_name = '{}_{}_{}_{}'.format(self.label_dict.get(self.ann_signal_label), x_col, y_col, self.source)
        label_list = copy.deepcopy(self.label_list)

        while self.ann_signal_label in label_list:
            label_list.remove(self.ann_signal_label)
        for b in label_list:
            fig_name = fig_name + '_' + str(b)

        plt.savefig(os.path.join(self.save_dir, fig_name + '.png'))
        plt.show()
        plt.close(fig)

def plot_bkg_ratio_imp_factor(imp_file_path_1, imp_file_path_2,x_col, y_col,source, save_dir, fig_name,y_ll=0, y_ul=1, x_ll=0.9, x_ul=1, x_log=False, y_log=False, **kwargs):

    gap=25
    label_size = 18
    text_dict = {
        'mc': 'MC test set\nMC training approach',
        'tb': 'Data test set\nData training approach'
    }

    imp_file_1=pd.read_csv(imp_file_path_1, usecols=[x_col, y_col])

    x_1=imp_file_1[x_col][::gap]
    y_1=imp_file_1[y_col][::gap]

    if 'y_scale' in kwargs.keys():
        y_1=y_1*kwargs.get('y_scale')

    imp_file_2 = pd.read_csv(imp_file_path_2, usecols=[x_col, y_col])

    x_2 = imp_file_2[x_col][::gap]
    y_2 = imp_file_2[y_col][::gap]

    if 'y_scale' in kwargs.keys():
        y_2 = y_2 * kwargs.get('y_scale')

    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )

    plt.text(0.1, 0.89, text_dict.get(source), fontsize=label_size, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )

    # plt.plot(x_1,
    #          y_1,
    #          'o--',
    #          markersize=8,
    #          linewidth=3,
    #          color='green',
    #          label='Pion rejection improvement',
    #          )
    print(x_2,y_2)
    plt.plot(x_2,
             y_2,
             '^--',
             markersize=10,
             linewidth=3,
             color='green',
             label='Electron background', )

    # make the plot readable

    plt.tick_params(labelsize=label_size, direction='in', length=5)

    plt.ylim(y_ll, y_ul)

    plt.xlim(x_ll, x_ul)
    if x_log:
        plt.xscale('log')
    else:
        plt.xticks(np.linspace(x_ll, x_ul, 5), fontsize=label_size)

        plt.minorticks_on()
        plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
        plt.xticks(np.linspace(x_ll, x_ul, 51), minor=True)

    if y_log:
        plt.yscale('log')
    else:
        plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=label_size-2)

        plt.minorticks_on()
        plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
        plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

    plt.xlabel('Pion efficiency ' + r'$(N_{S}^{sel.}/N_{S})$',
               fontsize=label_size - 2)
    plt.ylabel('Improvement '+r'$((R_e^\mathrm{ANN}-R_e^\mathrm{BDT})/R_e^\mathrm{BDT})$'+' [%]', fontsize=label_size - 2)
    plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.legend(bbox_to_anchor=(0.07, 0.8), bbox_transform=ax.transAxes, loc='upper left', fontsize=label_size - 2)
    if not os.path.exists(save_dir):

        os.mkdir(save_dir)

    plt.savefig(os.path.join(save_dir, fig_name + '.png'))
    plt.show()
    plt.close(fig)

def plot_bkg_ratio_imp_ratio(imp_file_path,  source, save_dir, fig_name, y_ll=0,
                              y_ul=1, x_ll=0.9, x_ul=1, x_log=False, y_log=False, **kwargs):

    gap = 25
    label_size = 18
    text_dict = {
        'mc': 'MC test set\nMC training approach',
        'tb': 'Data test set\nData training approach'
    }



    info = pd.read_csv(imp_file_path)
    x = info['ann_effi'].values
    bdt = info['bdt_bkg_ra'].values
    ann = info['ann_bkg_ra'].values

    x = x[::gap]
    bdt = np.round(bdt[::gap], decimals=1)
    ann = np.round(ann[::gap], decimals=1)


    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )

    plt.text(0.1, 0.89, text_dict.get(source), fontsize=label_size, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )



    plt.plot(x,
             np.round((ann)/bdt, decimals=1),
             '^--',
             markersize=10,
             linewidth=3,
             color='green',
             label='Electron background', )

    # make the plot readable

    plt.tick_params(labelsize=label_size, direction='in', length=5)

    plt.ylim(y_ll, y_ul)

    plt.xlim(x_ll-0.005, x_ul+0.005)
    if x_log:
        plt.xscale('log')
    else:
        plt.xticks(np.linspace(x_ll, x_ul, 5), fontsize=label_size)

        plt.minorticks_on()
        plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
        plt.xticks(np.linspace(x_ll, x_ul, 51), minor=True)

    if y_log:
        plt.yscale('log')
    else:
        plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=label_size - 2)

        plt.minorticks_on()
        plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
        plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

    plt.xlabel('Pion efficiency ' + r'$(N_{S}^{sel.}/N_{S})$',
               fontsize=label_size - 2)
    plt.ylabel('Bkg. rejection ratio ' + r'$(R_e^\mathrm{ANN}/R_e^\mathrm{BDT})$',
               fontsize=label_size - 2)
    plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.legend(bbox_to_anchor=(0.07, 0.8), bbox_transform=ax.transAxes, loc='upper left', fontsize=label_size - 2)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    plt.savefig(os.path.join(save_dir, fig_name + '.png'))
    plt.show()
    plt.close(fig)

def plot_granularity_cmp(imp_file_path_1, imp_file_path_2, imp_file_path_3, x_col, y_col,source, save_dir, fig_name,y_ll=0, y_ul=1, x_ll=0.9, x_ul=1, x_log=False, y_log=False, **kwargs):

    label_size = 18
    gap=24
    text_dict = {
        'mc': 'MC test set\nMC training approach',
        'tb': 'Data test set\nData training approach'
    }

    imp_file_1=pd.read_csv(imp_file_path_1, usecols=[x_col, y_col])

    x_1=imp_file_1[x_col][::gap]
    y_1=imp_file_1[y_col][::gap]

    if 'y_scale' in kwargs.keys():
        y_1=y_1*kwargs.get('y_scale')

    imp_file_2 = pd.read_csv(imp_file_path_2, usecols=[x_col, y_col])

    x_2 = imp_file_2[x_col][::gap]
    y_2 = imp_file_2[y_col][::gap]

    if 'y_scale' in kwargs.keys():
        y_2 = y_2 * kwargs.get('y_scale')

    imp_file_3 = pd.read_csv(imp_file_path_3, usecols=[x_col, y_col])
    x_3 = imp_file_3[x_col][::gap]
    y_3 = imp_file_3[y_col][::gap]

    if 'y_scale' in kwargs.keys():
        y_3 = y_3 * kwargs.get('y_scale')


    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )

    plt.text(0.1, 0.89, text_dict.get(source), fontsize=label_size, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )

    plt.plot(x_1,
             y_1,
             'o--',
             markersize=8,
             linewidth=3,

             label='Cell size: '+r'$\mathrm{4\times4\ cm^2}$',
             )

    plt.plot(x_2,
             y_2,
             '^--',
             markersize=8,
             linewidth=3,

             label='Cell size: '+r'$\mathrm{8\times8\ cm^2}$', )

    plt.plot(x_3,
             y_3,
             '*--',
             markersize=8,
             linewidth=3,

             label='Cell size: '+r'$\mathrm{12\times12\ cm^2}$', )

    # make the plot readable

    plt.tick_params(labelsize=label_size, direction='in', length=5)

    plt.ylim(y_ll, y_ul)

    plt.xlim(x_ll, x_ul)
    if x_log:
        plt.xscale('log')
    else:
        plt.xticks(np.linspace(x_ll, x_ul, 5), fontsize=label_size)

        plt.minorticks_on()
        plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
        plt.xticks(np.linspace(x_ll, x_ul, 51), minor=True)

    if y_log:
        plt.yscale('log')
    else:
        plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=label_size)

        plt.minorticks_on()
        plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
        plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

    plt.xlabel('Signal efficiency ' + r'$(N_{S}^{sel.}/N_{S})$',
               fontsize=label_size - 2)
    plt.ylabel('Bkg. efficiency ' + r'$(N_{B}^{sel.}/N_{B}$)', fontsize=label_size - 2)
    plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.legend(bbox_to_anchor=(0.01, 0.8), bbox_transform=ax.transAxes, loc='upper left', fontsize=label_size - 2)
    if not os.path.exists(save_dir):

        os.mkdir(save_dir)

    plt.savefig(os.path.join(save_dir, fig_name + '.png'))
    plt.show()
    plt.close(fig)


def plot_ep_cmp(ep_list_dict, var, source, index_list, label_list, save_dir, fig_name,y_ll=0, y_ul=1, x_ll=0.9, x_ul=1, x_log=False, y_log=False, y_scale=1,y_label='Bkg. rejection ' + r'$(N_{B}^{sel.}/N_{B}$)', **kwargs):

    label_size = 18
    text_dict = {
        'mc': 'MC test set\nMC training approach',
        'tb': 'Data test set\nData training approach'
    }

    var_dict=dict()

    for ep, file_path in ep_list_dict.items():

        file=pd.read_csv(file_path)

        var_dict[ep]=file[var][::25].values

    var_df=pd.DataFrame(var_dict, index=index_list)



    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()

    plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )

    plt.text(0.1, 0.89, text_dict.get(source), fontsize=label_size, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )

    for i, index in enumerate(index_list):

        plt.plot(list(var_df.columns),
                 var_df.values[i]*y_scale,
                 'o--',
                 markersize=8,
                 linewidth=3,

                 label=label_list[i],
                 )


    # make the plot readable

    plt.tick_params(labelsize=label_size, direction='in', length=5)

    plt.ylim(y_ll, y_ul)

    plt.xlim(x_ll, x_ul)
    if x_log:
        plt.xscale('log')
    else:
        plt.xticks([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], fontsize=label_size)



    if y_log:
        plt.yscale('log')
    else:
        plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=label_size)


        # plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

    plt.xlabel('Energy [GeV]',
               fontsize=label_size - 2)
    plt.ylabel(y_label, fontsize=label_size - 2)
    plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.legend(bbox_to_anchor=(0.95, 0.38), bbox_transform=ax.transAxes, loc='upper right', fontsize=label_size - 2)
    if not os.path.exists(save_dir):

        os.mkdir(save_dir)

    plt.savefig(os.path.join(save_dir, fig_name + '.png'))
    plt.show()
    plt.close(fig)

def plot_ann_bdt_compare(
        ann_file_1:dict,
        ann_file_2:dict,
        ann_file_3:dict,
        bdt_file_1:dict,
        bdt_file_2:dict,
        bdt_file_3:dict,
        save_dir:str,
        fig_name:str,
        ann_x_var:str,
        ann_y_var:str,
        bdt_x_var:str,
        bdt_y_var:str,
        source:str,
        x_ll:float,
        x_ul:float,
        y_ll:float,
        y_ul:float,
        y_scale:str,
        x_label:str,
        y_label:str,
        legend_x:float,
        legend_y:float,
        line_width:float,


):
    ann_1=pd.read_csv(ann_file_1.get('path'), usecols=[ann_x_var,ann_y_var])
    ann_2 = pd.read_csv(ann_file_2.get('path'), usecols=[ann_x_var,ann_y_var])
    ann_3 = pd.read_csv(ann_file_3.get('path'), usecols=[ann_x_var,ann_y_var])
    bdt_1 = pd.read_csv(bdt_file_1.get('path'), usecols=[bdt_x_var,bdt_y_var])
    bdt_2 = pd.read_csv(bdt_file_2.get('path'), usecols=[bdt_x_var,bdt_y_var])
    bdt_3 = pd.read_csv(bdt_file_3.get('path'), usecols=[bdt_x_var,bdt_y_var])

    ann_x_1 = ann_1[ann_x_var]
    ann_x_2 = ann_2[ann_x_var]
    ann_x_3 = ann_3[ann_x_var]
    bdt_x_1 = bdt_1[bdt_x_var]
    bdt_x_2 = bdt_2[bdt_x_var]
    bdt_x_3 = bdt_3[bdt_x_var]

    ann_y_1 = ann_1[ann_y_var]
    ann_y_2 = ann_2[ann_y_var]
    ann_y_3 = ann_3[ann_y_var]
    bdt_y_1 = bdt_1[bdt_y_var]
    bdt_y_2 = bdt_2[bdt_y_var]
    bdt_y_3 = bdt_3[bdt_y_var]

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

    plt.text(0.1, 0.89, text_dict.get(source), fontsize=label_size, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes,
             )

    plt.plot(ann_x_1,
             ann_y_1, ann_file_1.get('style'), markersize=6,
             color=ann_file_1.get('color'), label=ann_file_1.get('label'),
             linewidth=line_width,
             )

    plt.plot(ann_x_2,
             ann_y_2, ann_file_2.get('style'), markersize=6,
             color=ann_file_2.get('color'), label=ann_file_2.get('label'),
             linewidth=line_width,)

    plt.plot(ann_x_3,
             ann_y_3, ann_file_3.get('style'), markersize=6,
             color=ann_file_3.get('color'), label=ann_file_3.get('label'),
             linewidth=line_width,)

    plt.plot(bdt_x_1,
             bdt_y_1, bdt_file_1.get('style'), markersize=6,
             color=bdt_file_1.get('color'), label=bdt_file_1.get('label'),
             linewidth=line_width-1,
             )

    plt.plot(bdt_x_2,
             bdt_y_2, bdt_file_2.get('style'), markersize=6,
             color=bdt_file_2.get('color'), label=bdt_file_2.get('label'),
             linewidth=line_width-1,)

    plt.plot(bdt_x_3,
             bdt_y_3, bdt_file_3.get('style'), markersize=6,
             color=bdt_file_3.get('color'), label=bdt_file_3.get('label'),
             linewidth=line_width-1,)

    # make the plot readable

    plt.tick_params(labelsize=label_size, direction='in', length=5)

    plt.ylim(y_ll, y_ul)
    plt.xlim(x_ll, x_ul)
    plt.xticks(np.linspace(x_ll, x_ul, round(11 - 100 * (1 - x_ul))), fontsize=label_size)


    plt.minorticks_on()
    plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
    plt.xticks(np.linspace(0.9, x_ul, round(500 * (x_ul - 0.9) + 1)), minor=True)

    plt.yscale(y_scale)
    if y_scale=='linear':
        plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=label_size)
        plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

    plt.xlabel(xlabel=x_label,
               fontsize=label_size)
    plt.ylabel(ylabel=y_label,
               fontsize=label_size)

    plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.legend(bbox_to_anchor=(legend_x, legend_y), bbox_transform=ax.transAxes, loc='upper right', fontsize=label_size - 4)

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(os.path.join(save_dir, fig_name + '.png'))
    plt.show()
    plt.close(fig)





def main_1():
    plot_ann_bdt_compare(
        ann_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/ann_detailed_s_0_b_1_2.csv',
            'style': '^-',
            'color': 'red',
            'label': r'$ANN, signal: \mu, bkg.:e, \pi$'},
        ann_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/ann_detailed_s_0_b_1.csv',
            'style': '^-',
            'color': 'blue',
            'label': r'$ANN, signal: \mu, bkg.: e$'},
        ann_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/ann_detailed_s_0_b_2.csv',
            'style': '^-',
            'color': 'green',
            'label': r'$ANN, signal: \mu, bkg.: \pi$'},

        bdt_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/bdt_detailed_s_0_b_1_2.csv',
            'style': 'o--',
            'color': 'red',
            'label': r'$BDT, signal: \mu, bkg.:e, \pi$'},
        bdt_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/bdt_detailed_s_0_b_1.csv',
            'style': 'o--',
            'color': 'blue',
            'label': r'$BDT, signal: \mu, bkg.: e$'},
        bdt_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/bdt_detailed_s_0_b_2.csv',
            'style': 'o--',
            'color': 'green',
            'label': r'$BDT, signal: \mu, bkg.: \pi$'},
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1',
        fig_name='muon_compare',
        ann_x_var='ann_effi',
        ann_y_var='ann_bkg_ratio',
        bdt_x_var='bdt_effi',
        bdt_y_var='bdt_bkg_ratio',
        source='mc',
        x_ll=0.9,
        x_ul=1,
        y_ll=1,
        y_ul=30000,
        y_scale='log',
        x_label='Muon efficiency' + r'$(N_{S}^{sel.}/N_{S})$',
        y_label='Bkg. rejection ' + r'$(N_{B}/N_{B}^{sel.}$)',
        legend_x=0.6,
        legend_y=0.5,
        line_width=4,
    )

    plot_ann_bdt_compare(
        ann_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/ann_detailed_s_1_b_0_2.csv',
            'style': '^-',
            'color': 'red',
            'label': r'$ANN, signal: e, bkg.: \mu, \pi$'},
        ann_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/ann_detailed_s_1_b_0.csv',
            'style': '^-',
            'color': 'blue',
            'label': r'$ANN, signal: e, bkg.: \mu$'},
        ann_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/ann_detailed_s_1_b_2.csv',
            'style': '^-',
            'color': 'green',
            'label': r'$ANN, signal: e, bkg.: \pi$'},

        bdt_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/bdt_detailed_s_1_b_0_2.csv',
            'style': 'o--',
            'color': 'red',
            'label': r'$BDT, signal: e, bkg.: \mu, \pi$'},
        bdt_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/bdt_detailed_s_1_b_0.csv',
            'style': 'o--',
            'color': 'blue',
            'label': r'$BDT, signal: e, bkg.: \mu$'},
        bdt_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/bdt_detailed_s_1_b_2.csv',
            'style': 'o--',
            'color': 'green',
            'label': r'$BDT, signal: e, bkg.: \pi$'},
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1',
        fig_name='electron_compare',
        ann_x_var='ann_effi',
        ann_y_var='ann_bkg_ratio',
        bdt_x_var='bdt_effi',
        bdt_y_var='bdt_bkg_ratio',
        source='mc',
        x_ll=0.9,
        x_ul=1,
        y_ll=1,
        y_ul=30000,
        y_scale='log',
        x_label='Electron efficiency' + r'$(N_{S}^{sel.}/N_{S})$',
        y_label='Bkg. rejection ' + r'$(N_{B}/N_{B}^{sel.}$)',
        legend_x=0.6,
        legend_y=0.5,
        line_width=4,
    )

    plot_ann_bdt_compare(
        ann_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/ann_detailed_s_2_b_0_1.csv',
            'style': '^-',
            'color': 'red',
            'label': r'$ANN, signal: \pi, bkg.: \mu, e$'},
        ann_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/ann_detailed_s_2_b_0.csv',
            'style': '^-',
            'color': 'blue',
            'label': r'$ANN, signal: \pi, bkg.: \mu$'},
        ann_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/ann_detailed_s_2_b_1.csv',
            'style': '^-',
            'color': 'green',
            'label': r'$ANN, signal: \pi, bkg.: e$'},

        bdt_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/bdt_detailed_s_2_b_0_1.csv',
            'style': 'o--',
            'color': 'red',
            'label': r'$BDT, signal: \pi, bkg.: \mu, e$'},
        bdt_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/bdt_detailed_s_2_b_0.csv',
            'style': 'o--',
            'color': 'blue',
            'label': r'$BDT, signal: \pi, bkg.: \mu$'},
        bdt_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/bdt_detailed_s_2_b_1.csv',
            'style': 'o--',
            'color': 'green',
            'label': r'$BDT, signal: \pi, bkg.: e$'},
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1',
        fig_name='pion_compare',
        ann_x_var='ann_effi',
        ann_y_var='ann_bkg_ratio',
        bdt_x_var='bdt_effi',
        bdt_y_var='bdt_bkg_ratio',
        source='mc',
        x_ll=0.9,
        x_ul=1,
        y_ll=1,
        y_ul=100000,
        y_scale='log',
        x_label='Pion efficiency' + r'$(N_{S}^{sel.}/N_{S})$',
        y_label='Bkg. rejection ' + r'$(N_{B}/N_{B}^{sel.}$)',
        legend_x=0.6,
        legend_y=0.5,
        line_width=4,
    )


def main_2():
    plot_ann_bdt_compare(
        ann_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/ann_detailed_s_0_b_1_2.csv',
            'style': '-',
            'color': 'red',
            'label': r'$ANN, signal: \mu, bkg.:e, \pi$'},
        ann_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/ann_detailed_s_0_b_1.csv',
            'style': '-',
            'color': 'blue',
            'label': r'$ANN, signal: \mu, bkg.: e$'},
        ann_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/ann_detailed_s_0_b_2.csv',
            'style': '-',
            'color': 'green',
            'label': r'$ANN, signal: \mu, bkg.: \pi$'},

        bdt_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/bdt_detailed_s_0_b_1_2.csv',
            'style': '--',
            'color': 'red',
            'label': r'$BDT, signal: \mu, bkg.:e, \pi$'},
        bdt_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/bdt_detailed_s_0_b_1.csv',
            'style': '--',
            'color': 'blue',
            'label': r'$BDT, signal: \mu, bkg.: e$'},
        bdt_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/bdt_detailed_s_0_b_2.csv',
            'style': '--',
            'color': 'green',
            'label': r'$BDT, signal: \mu, bkg.: \pi$'},
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1',
        fig_name='muon_compare_effi',
        ann_x_var='ann_effi',
        ann_y_var='ann_bkg_effi',
        bdt_x_var='bdt_effi',
        bdt_y_var='bdt_bkg_effi',
        source='mc',
        x_ll=0.9,
        x_ul=1,
        y_ll=0.000001,
        y_ul=1,
        y_scale='log',
        x_label='Muon efficiency' + r'$(N_{S}^{sel.}/N_{S})$',
        y_label='Bkg. efficiency ' + r'$(N_{B}^{sel.}/N_{B})$',
        legend_x=0.6,
        legend_y=0.4,
        line_width=4,
    )

    plot_ann_bdt_compare(
        ann_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/ann_detailed_s_1_b_0_2.csv',
            'style': '-',
            'color': 'red',
            'label': r'$ANN, signal: e, bkg.: \mu, \pi$'},
        ann_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/ann_detailed_s_1_b_0.csv',
            'style': '-',
            'color': 'blue',
            'label': r'$ANN, signal: e, bkg.: \mu$'},
        ann_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/ann_detailed_s_1_b_2.csv',
            'style': '-',
            'color': 'green',
            'label': r'$ANN, signal: e, bkg.: \pi$'},

        bdt_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/bdt_detailed_s_1_b_0_2.csv',
            'style': '--',
            'color': 'red',
            'label': r'$BDT, signal: e, bkg.: \mu, \pi$'},
        bdt_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/bdt_detailed_s_1_b_0.csv',
            'style': '--',
            'color': 'blue',
            'label': r'$BDT, signal: e, bkg.: \mu$'},
        bdt_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/bdt_detailed_s_1_b_2.csv',
            'style': '--',
            'color': 'green',
            'label': r'$BDT, signal: e, bkg.: \pi$'},
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1',
        fig_name='electron_compare_effi',
        ann_x_var='ann_effi',
        ann_y_var='ann_bkg_effi',
        bdt_x_var='bdt_effi',
        bdt_y_var='bdt_bkg_effi',
        source='mc',
        x_ll=0.9,
        x_ul=1,
        y_ll=0.000001,
        y_ul=1,
        y_scale='log',
        x_label='Electron efficiency' + r'$(N_{S}^{sel.}/N_{S})$',
        y_label='Bkg. efficiency ' + r'$(N_{B}^{sel.}/N_{B})$',
        legend_x=0.6,
        legend_y=0.4,
        line_width=4,
    )

    plot_ann_bdt_compare(
        ann_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/ann_detailed_s_2_b_0_1.csv',
            'style': '-',
            'color': 'red',
            'label': r'$ANN, signal: \pi, bkg.: \mu, e$'},
        ann_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/ann_detailed_s_2_b_0.csv',
            'style': '-',
            'color': 'blue',
            'label': r'$ANN, signal: \pi, bkg.: \mu$'},
        ann_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/ann_detailed_s_2_b_1.csv',
            'style': '-',
            'color': 'green',
            'label': r'$ANN, signal: \pi, bkg.: e$'},

        bdt_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/bdt_detailed_s_2_b_0_1.csv',
            'style': '--',
            'color': 'red',
            'label': r'$BDT, signal: \pi, bkg.: \mu, e$'},
        bdt_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/bdt_detailed_s_2_b_0.csv',
            'style': '--',
            'color': 'blue',
            'label': r'$BDT, signal: \pi, bkg.: \mu$'},
        bdt_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/bdt_detailed_s_2_b_1.csv',
            'style': '--',
            'color': 'green',
            'label': r'$BDT, signal: \pi, bkg.: e$'},
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1',
        fig_name='pion_compare_effi',
        ann_x_var='ann_effi',
        ann_y_var='ann_bkg_effi',
        bdt_x_var='bdt_effi',
        bdt_y_var='bdt_bkg_effi',
        source='mc',
        x_ll=0.9,
        x_ul=1,
        y_ll=0.000001,
        y_ul=1,
        y_scale='log',
        x_label='Pion efficiency' + r'$(N_{S}^{sel.}/N_{S})$',
        y_label='Bkg. efficiency ' + r'$(N_{B}^{sel.}/N_{B})$',
        legend_x=0.95,
        legend_y=0.4,
        line_width=4,
    )

def main_ana(ckp):
    num = 500
    for signal in range(3):
        cmp = Compare(
            bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_signal_{}_md_100_nt_100_v3/eval.csv'.format(
                signal),
            ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/PIDTags/TV/imgs_ANN.root'.format(ckp),
            raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_1_1/TV/labels.npy',
            save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/compare_{}_v1'.format(ckp,
                signal),
            ann_threshold_lists=np.linspace(0, 0.99999, 10000),
            bdt_threshold_lists=np.linspace(0, 1, 2000),
            ann_signal_label=signal,
            source='mc',
            n_classes=4
        )
        cmp.filter_label(label_list=[0, 1, 2])
        cmp.get_bdt_ns_nb()
        # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
        # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
        cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=30000)
        cmp.export_info()
        cmp.export_improvement_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1])

        label_list = [0, 1, 2]
        label_list.remove(signal)

        for b in label_list:
            cmp = Compare(
                bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_signal_{}_md_100_nt_100_v3/eval.csv'.format(
                    signal),
                ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/PIDTags/TV/imgs_ANN.root'.format(ckp),
                raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_1_1/TV/labels.npy',
                save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/compare_{}_v1'.format(ckp,
                    signal),
                ann_threshold_lists=np.linspace(0, 0.99999, 10000),
                bdt_threshold_lists=np.linspace(0, 1, 2000),
                ann_signal_label=signal,
                source='mc',
                n_classes=4
            )
            cmp.filter_label(label_list=[signal, b])
            cmp.get_bdt_ns_nb()
            # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
            # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
            cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=30000)
            cmp.export_info()
            cmp.export_improvement_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1])

def main_ana_e_pi(ckp):
    #
    label_dict={
        0: 'Electron',
        1: 'Pion',
    }
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0915_mc_s_0_b_1_md_100_nt_100_var_12/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/PIDTags/Test/imgs_ANN.root'.format(ckp),
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/mc_0720_e_pi/Test/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/compare_0_v1'.format(ckp),
    #     ann_threshold_lists=np.linspace(0, 1, 10000),
    #     bdt_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=0,
    #     source='mc',
    #     n_classes=2,
    #     label_dict=label_dict
    # )
    # cmp.filter_label(label_list=[0,1])
    # cmp.get_bdt_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ll=0, x_ul=1, y_ll=1, y_ul=3000000)
    # cmp.plot_bkg_effi_compare(x_ul=1, y_ll=0.0001, x_ll=0, y_ul=1)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1])

    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0915_mc_s_1_b_0_md_100_nt_100_var_12/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/PIDTags/Test/imgs_ANN.root'.format(ckp),
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/mc_0720_e_pi/Test/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/compare_1_v1'.format(ckp),
    #     ann_threshold_lists=np.power(np.linspace(0, 1, 10000), 6),
    #     bdt_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=1,
    #     source='mc',
    #     n_classes=2,
    #     label_dict=label_dict,
    #     label_list=[0,1]
    # )
    # # cmp.filter_label()
    # cmp.get_bdt_ns_nb()
    # cmp.get_ann_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ll=0, x_ul=1, y_ll=1, y_ul=10000000)
    # cmp.plot_bkg_effi_compare(x_ul=1, y_ll=0.00001, x_ll=0.9, y_ul=1, x_log=False)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1])

    ann_scores=read_ann_score(
        file_pid_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/PIDTags/Test/imgs_ANN.root'.format(ckp),
        n_classes=2,
    )

    labels=np.load('/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/mc_0720_e_pi_block_1_1/Test/labels.npy')
    ann_score_2 = read_ann_score(
        file_pid_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/PIDTags/Test_2/imgs_ANN.root'.format(ckp),
        n_classes=2,
    )
    labels_2 = np.load('/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/mc_0720_e_pi_2/Test/labels.npy')

    ann_scores=np.concatenate([ann_scores,ann_score_2], axis=0)
    labels=np.concatenate([labels, labels_2], axis=0)

    ann_acc = np.sum((np.argmax(ann_scores, axis=1) == labels) != 0) / len(labels)
    print('ann_acc: {}'.format(round(100*ann_acc, 3)))
    # bdt_eval= pd.read_csv('/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0915_mc_s_0_b_1_md_100_nt_100_var_12/eval.csv')


    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0915_mc_s_0_b_1_md_100_nt_100_var_12/eval.csv',
    #     ann_scores_path=None,
    #     raw_labels_path=None,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/compare_0_v2'.format(ckp),
    #     ann_threshold_lists=np.unique(np.sort(ann_scores[:, 0])),
    #     bdt_threshold_lists=np.unique(np.sort(bdt_eval['predictions'].values)),
    #     ann_signal_label=0,
    #     source='mc',
    #     n_classes=2,
    #     label_dict=label_dict,
    #     ann_scores=ann_scores,
    #     raw_labels=labels,
    #     label_list=[0, 1]
    # )
    # # cmp.filter_label()
    # cmp.get_bdt_ns_nb()
    # cmp.get_ann_ns_nb()
    # # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ll=0.9, x_ul=1, y_ll=1, y_ul=30000)
    # cmp.plot_bkg_effi_compare(x_ul=1, y_ll=0.00001, x_ll=0.9, y_ul=1, x_log=False)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=np.linspace(0.99, 0.95, 101))

    bdt_eval = pd.read_csv(
        '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0915_mc_s_1_b_0_md_100_nt_100_var_12/eval.csv')
    bdt_scores=bdt_eval['predictions'].values
    bdt_scores=bdt_scores.reshape(-1,1)
    bdt_label=bdt_eval['raw_labels'].values
    bdt_scores=np.hstack([1-bdt_scores, bdt_scores])

    bdt_acc = np.sum((np.argmax(bdt_scores, axis=1) == bdt_label) != 0) / len(bdt_label)
    print('bdt_acc: {}'.format(round(100 * bdt_acc, 3)))
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0915_mc_s_1_b_0_md_100_nt_100_var_12/eval.csv',
    #     ann_scores_path=None,
    #     raw_labels_path=None,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/compare_1_v2'.format(ckp),
    #     ann_threshold_lists=np.unique(np.sort(ann_scores[:, 1])),
    #     bdt_threshold_lists=np.unique(np.sort(bdt_eval['predictions'].values)),
    #     ann_signal_label=1,
    #     source='mc',
    #     n_classes=2,
    #     label_dict=label_dict,
    #     ann_scores=ann_scores,
    #     raw_labels=labels,
    #     label_list=[0, 1]
    # )
    # cmp.filter_label()
    # cmp.get_bdt_ns_nb()
    # cmp.get_ann_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ll=0.95, x_ul=1, y_ll=1, y_ul=10000000,
    #                            ann_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_draft_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/ann_detailed_s_1_b_0.csv',
    #                            bdt_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_draft_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/bdt_detailed_s_1_b_0.csv')
    # cmp.plot_bkg_effi_compare(x_ul=1, y_ll=0.00001, x_ll=0.9, y_ul=1, x_log=False)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=np.linspace(0.99, 0.95, 101))

def main_ana_e_pi_16m(ann_ckp,
                      bdt_eval_path):
    #
    label_dict={
        0: 'Electron',
        1: 'Pion',
    }


    ann_df=pd.read_csv(os.path.join(ann_ckp, 'ANA/imgs_ANN.csv'),   )
    ann_scores=ann_df[['ANN_e','ANN_pi']].values

    labels=ann_df['particle_label'].values


    ann_acc = np.sum((np.argmax(ann_scores, axis=1) == labels) != 0) / len(labels)
    print('ann_acc: {}'.format(round(100*ann_acc, 3)))


    bdt_eval = pd.read_csv(bdt_eval_path)
    bdt_scores=bdt_eval['predictions'].values
    bdt_scores=bdt_scores.reshape(-1,1)
    bdt_label=bdt_eval['raw_labels'].values
    bdt_scores=np.hstack([1-bdt_scores, bdt_scores])

    bdt_acc = np.sum((np.argmax(bdt_scores, axis=1) == bdt_label) != 0) / len(bdt_label)
    print('bdt_acc: {}'.format(round(100 * bdt_acc, 3)))
    cmp = Compare(
        bdt_eval_path=bdt_eval_path,
        ann_scores_path=None,
        raw_labels_path=None,
        save_dir=os.path.join(ann_ckp, 'ANA/compare'),
        ann_threshold_lists=np.unique(np.sort(ann_scores[:, 1])),
        bdt_threshold_lists=np.unique(np.sort(bdt_eval['predictions'].values)),
        ann_signal_label=1,
        source='mc',
        n_classes=2,
        label_dict=label_dict,
        ann_scores=ann_scores,
        raw_labels=labels,
        label_list=[0, 1]
    )
    cmp.filter_label()
    cmp.get_bdt_ns_nb()
    cmp.get_ann_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ll=0.95, x_ul=1, y_ll=1, y_ul=10000000,
    #                           )
    # cmp.plot_bkg_effi_compare(x_ul=1, y_ll=0.00001, x_ll=0.9, y_ul=1, x_log=False)
    cmp.export_info()
    cmp.export_improvement_info(effi_points=np.linspace(0.995, 0.95, 111))
def main_sep(ckp, ep_list):

    label_dict = {
        0: 'Electron',
        1: 'Pion',
    }

    dir_list = [str(i) + 'GeV' for i in ep_list]

    for dir in dir_list:

        ann_scores_path= '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/PIDTags/sep_e_pi/{}/imgs_ANN.root'.format(ckp,dir)
        label_path= '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/sep_e_pi/{}/labels.npy'.format(dir)

        ann_scores=read_ann_score(ann_scores_path,
                                  n_classes=2)


        cmp = Compare(
            bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0915_mc_s_0_b_1_md_100_nt_100_var_12/eval.csv',
            ann_scores_path= ann_scores_path,
            raw_labels_path=label_path,
            save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/{}'.format(ckp, dir),
            ann_threshold_lists=np.sort(ann_scores[:,0]),
            bdt_threshold_lists=np.linspace(0, 1, 1000),
            ann_signal_label=0,
            source='mc',
            n_classes=2,
            label_dict=label_dict,
            label_list=[0,1]
        )
        # cmp.filter_label()
        cmp.get_ann_ns_nb()
        cmp.get_bdt_ns_nb()
        cmp.plot_bkg_ratio_compare(x_ll=0.95, x_ul=1, y_ll=1, y_ul=10000,)
        # cmp.export_info()
        cmp.export_improvement_info(effi_points=np.linspace(0.99, 0.95, 101))

        cmp = Compare(
            bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0915_mc_s_1_b_0_md_100_nt_100_var_12/eval.csv',
            ann_scores_path=ann_scores_path,
            raw_labels_path=label_path,
            save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/{}'.format(ckp, dir),
            ann_threshold_lists=np.sort(ann_scores[:, 1]),
            bdt_threshold_lists=np.linspace(0, 1, 1000),
            ann_signal_label=1,
            source='mc',
            n_classes=2,
            label_dict=label_dict,
            label_list=[0, 1]
        )
        # cmp.filter_label()
        cmp.get_ann_ns_nb()
        cmp.get_bdt_ns_nb()
        cmp.plot_bkg_ratio_compare(x_ll=0.95, x_ul=1, y_ll=1, y_ul=10000000, )
        # cmp.export_info()
        cmp.export_improvement_info(effi_points=np.linspace(0.99, 0.95, 101))

def main_granularity_cmp():


    # plot_granularity_cmp(
    #     imp_file_path_1='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_0_v2/ann_info_s_0_b_1.csv',
    #     imp_file_path_2='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_2_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_0_v2/ann_info_s_0_b_1.csv',
    #     imp_file_path_3='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_3_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_0_v2/ann_info_s_0_b_1.csv',
    #     x_col='ann_effi',
    #     y_col='ann_bkg_ra',
    #     x_ll=0.95,
    #     x_ul=0.99,
    #     y_ll=100,
    #     y_ul=10000,
    #     y_log=True,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/Fig',
    #     fig_name='bkg_ratio_granu_cmp_0',
    #     source='mc'
    # )
    #
    # plot_granularity_cmp(
    #     imp_file_path_1='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/ann_info_s_1_b_0.csv',
    #     imp_file_path_2='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_2_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/ann_info_s_1_b_0.csv',
    #     imp_file_path_3='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_3_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/ann_info_s_1_b_0.csv',
    #     x_col='ann_effi',
    #     y_col='ann_bkg_ra',
    #     x_ll=0.95,
    #     x_ul=0.99,
    #     y_ll=100,
    #     y_ul=1000000,
    #     y_log=True,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/Fig',
    #     fig_name='bkg_ratio_granu_cmp_1',
    #     source='mc'
    # )

    plot_granularity_cmp(
        imp_file_path_1='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_0_v2/ann_detailed_s_0_b_1.csv',
        imp_file_path_2='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_2_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_0_v2/ann_detailed_s_0_b_1.csv',
        imp_file_path_3='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_3_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_0_v2/ann_detailed_s_0_b_1.csv',
        x_col='ann_effi',
        y_col='ann_bkg_ratio',
        x_ll=0.95,
        x_ul=0.99,
        y_ll=100,
        y_ul=10000,
        y_log=True,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/Fig',
        fig_name='bkg_ratio_granu_cmp_0_detailed',
        source='mc'
    )

    plot_granularity_cmp(
        imp_file_path_1='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/ann_detailed_s_1_b_0.csv',
        imp_file_path_2='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_2_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/ann_detailed_s_1_b_0.csv',
        imp_file_path_3='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_3_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/ann_detailed_s_1_b_0.csv',
        x_col='ann_effi',
        y_col='ann_bkg_ratio',
        x_ll=0.95,
        x_ul=0.99,
        y_ll=100,
        y_ul=1000000,
        y_log=True,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/Fig',
        fig_name='bkg_ratio_granu_cmp_1_detailed',
        source='mc'
    )

def main_ep():
    ep_dict = dict()

    for ep in [5, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
        file_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/{}GeV/ann_info_s_0_b_1.csv'.format(
            ep)
        ep_dict[ep] = file_path

    index_list = np.linspace(0.95, 0.99, 5)
    label_list = [str(int(100 * i)) + '% ' + r'$\varepsilon_{e}$' for i in index_list]
    plot_ep_cmp(ep_list_dict=ep_dict,
                var='ann_bkg_ra',
                index_list=index_list,
                label_list=label_list,
                x_ll=5,
                x_ul=120,
                y_ll=0,
                y_ul=1200,
                y_log=False,
                save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/Fig',
                fig_name='bkg_ratio_ep_0',
                source='mc'
                )


    ep_dict=dict()

    for ep in [5, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
        file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/{}GeV/ann_info_s_1_b_0.csv'.format(ep)
        ep_dict[ep]=file_path

    index_list=np.linspace(0.95, 0.99,5)
    label_list=[str(int(100*i))+ '% '+ r'$\varepsilon_{\pi}$' for i in index_list]
    plot_ep_cmp(ep_list_dict=ep_dict,
                var='ann_bkg_ra',
                index_list=index_list,
                label_list=label_list,
                x_ll=5,
                x_ul=120,
                y_ll=100,
                y_ul=1000000,
                y_log=True,
                save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/Fig',
                fig_name='bkg_ratio_ep_1',
                source='mc'
                )

    ep_dict = dict()

    for ep in [5, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
        file_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/{}GeV/ann_info_s_1_b_0.csv'.format(
            ep)
        ep_dict[ep] = file_path

    index_list = np.linspace(0.95, 0.99, 5)
    label_list = [str(int(100 * i)) + '% ' + r'$\varepsilon_{\pi}$' for i in index_list]
    plot_ep_cmp(ep_list_dict=ep_dict,
                var='ann_puri',
                index_list=index_list,
                label_list=label_list,
                x_ll=5,
                x_ul=120,
                y_ll=99.5,
                y_ul=100.5,
                y_log=False,
                y_scale=100,
                save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/Fig',
                fig_name='bkg_puri_ep_1',
                source='mc',
                y_label='Pion purity [%]',
                )


if __name__ == '__main__':


    # TODO draft v1
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_mc/Test/0720_mc_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version/Test/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0720_mc_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_v1/ANA/compare',
    #     ann_threshold_lists=np.linspace(0, 1, 10000),
    #     bdt_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=2,
    #     source='mc',
    #     n_classes=4
    # )
    # cmp.filter_label(label_list=[0,1,2])
    # cmp.get_bdt_ns_nb()
    # # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=[0.90, 0.92, 0.94, 0.96, 0.98, 0.99][::-1])
    #
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_ckv_fd_0720_beam/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_tb/Test/0720_tb_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/ckv_fd_0720/Test/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0720_tb_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_v1/ANA/compare',
    #     ann_threshold_lists=np.linspace(0, 1, 10000),
    #     bdt_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=2,
    #     source='tb',
    #     n_classes=4
    # )
    # cmp.filter_label(label_list=[0, 1, 2])
    # cmp.get_bdt_ns_nb()
    # # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=[0.90, 0.92, 0.94, 0.96, 0.98, 0.99][::-1])

    # # TODO draft v1.3
    # num=500
    # for signal in range(3):
    #     cmp = Compare(
    #         bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_signal_{}_md_100_nt_100_v3/eval.csv'.format(signal),
    #         ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res18_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_4_l_gamma_0.1_step_50_st_True_b_1_1_v1/ANA/PIDTags/TV/imgs_ANN.root',
    #         raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_1_1/TV/labels.npy',
    #         save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res18_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_4_l_gamma_0.1_step_50_st_True_b_1_1_v1/ANA/compare_{}_v3'.format(signal),
    #         ann_threshold_lists=np.linspace(0, 0.99999, 10000),
    #         bdt_threshold_lists=np.linspace(0, 1, 2000),
    #         ann_signal_label=signal,
    #         source='mc',
    #         n_classes=4
    #     )
    #     cmp.filter_label(label_list=[0,1,2])
    #     cmp.get_bdt_ns_nb()
    #     # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    #     # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    #     # cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=30000)
    #     cmp.export_info()
    #     cmp.export_improvement_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1])
    #
    #     label_list=[0,1,2]
    #     label_list.remove(signal)
    #
    #     for b in label_list:
    #         cmp = Compare(
    #             bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_signal_{}_md_100_nt_100_v3/eval.csv'.format(
    #                 signal),
    #             ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res18_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_4_l_gamma_0.1_step_50_st_True_b_1_1_v1/ANA/PIDTags/TV/imgs_ANN.root',
    #             raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_1_1/TV/labels.npy',
    #             save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res18_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_4_l_gamma_0.1_step_50_st_True_b_1_1_v1/ANA/compare_{}_v3'.format(
    #                 signal),
    #             ann_threshold_lists=np.linspace(0, 0.99999, 10000),
    #             bdt_threshold_lists=np.linspace(0, 1, 2000),
    #             ann_signal_label=signal,
    #             source='mc',
    #             n_classes=4
    #         )
    #         cmp.filter_label(label_list=[signal, b])
    #         cmp.get_bdt_ns_nb()
    #         # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    #         # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    #         # cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=30000)
    #         cmp.export_info()
    #         cmp.export_improvement_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1])

    # TODO draft v1.4
    # main_ana(ckp='0901_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_4_l_gamma_1.0_step_100_st_True_b_1_1_f_k_3_f_s_2_f_p_3_v1')
    #
    # main_1()
    # main_2()



    #
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_ckv_fd_0720_beam_md_100_nt_100_v3/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_tb/TV/0728_tb_res18_epoch_200_lr_1e-05_batch32_optim_Adam_classes_4_l_gamma_0.5_step_10v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/ckv_fd_0720/TV/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_tb_res18_epoch_200_lr_1e-05_batch32_optim_Adam_classes_4_l_gamma_0.5_step_10v1/ANA/compare_v3',
    #     ann_threshold_lists=np.linspace(0, 0.99999, 50000),
    #     bdt_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=2,
    #     source='tb',
    #     n_classes=4
    # )
    # cmp.filter_label(label_list=[0, 1, 2])
    # cmp.get_bdt_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=15000)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=[0.90, 0.95, 0.99][::-1])

    # # TODO draft v2 no noise
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_ckv_fd_0720_no_noise_beam/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_tb/Test/0728_tb_res18_epoch_200_lr_1e-06_batch64_optim_SGD_classes_3_v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/ckv_fd_0720_no_noise/Test/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_tb_res18_epoch_200_lr_1e-06_batch64_optim_SGD_classes_3_v1/ANA/compare',
    #     ann_threshold_lists=np.linspace(0, 1, 800),
    #     bdt_threshold_lists=np.linspace(0, 1, 500),
    #     ann_signal_label=2,
    #     n_classes=3,
    #     source='tb'
    # )
    # cmp.filter_label(label_list=[0, 1, 2])
    # cmp.get_bdt_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=[0.91, 0.93, 0.95, 0.97, 0.99][::-1])
    #
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_no_noise_mc_md_10_nt_10_v4/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_mc/TV/0728_mc_res34_epoch_200_lr_0.001_batch32_optim_Adam_classes_3_l_gamma_0.1_step_100v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_no_noise/TV/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res34_epoch_200_lr_0.001_batch32_optim_Adam_classes_3_l_gamma_0.1_step_100v1/ANA/compare',
    #     ann_threshold_lists=np.linspace(0, 0.99999, 10000),
    #     bdt_threshold_lists=np.linspace(0, 1, 1000),
    #     ann_signal_label=2,
    #     n_classes=3,
    #     source='mc'
    # )
    # cmp.filter_label(label_list=[0, 1, 2])
    # cmp.get_bdt_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=10000)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=[0.91, 0.93, 0.95, 0.97, 0.99][::-1])

    # label_dict = {
    #     0: 'Electron',
    #     1: 'Pion',
    # }
    #
    # ckp = '0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1'
    #
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0915_mc_s_0_b_1_md_100_nt_100_var_12/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/PIDTags/Test/imgs_ANN.root'.format(ckp),
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/mc_0720_e_pi/Test/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/compare_0_v2'.format(ckp),
    #     ann_threshold_lists=np.power(np.linspace(0, 1, 10000), 6),
    #     bdt_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=0,
    #     source='mc',
    #     n_classes=2,
    #     label_dict=label_dict,
    #     label_list=[0, 1]
    # )
    # # cmp.filter_label()
    #
    # cmp.plot_bkg_ratio_compare(x_ll=0.95, x_ul=1, y_ll=1, y_ul=10000,
    #                            ann_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_0_v2/ann_detailed_s_0_b_1.csv',
    #                            bdt_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_0_v2/bdt_detailed_s_0_b_1.csv')
    # cmp.plot_bkg_effi_compare(x_ll=0.9, x_ul=1, y_ll=0.00001, y_ul=1,x_log=False, y_log=True,
    #                           ann_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_0_v2/ann_detailed_s_0_b_1.csv',
    #                           bdt_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_0_v2/bdt_detailed_s_0_b_1.csv'
    #                           )
    # cmp.plot_bkg_ratio_imp_factor(
    #     imp_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_0_v2/ann_info_s_0_b_1.csv',
    #     x_col='ann_effi',
    #     y_col='bkg_ra_imp',
    #     x_ll=0.95,
    #     x_ul=0.99,
    #     y_ll=0,
    #     y_ul=4,
    #     y_scale=1 / 100,
    #
    # )
    # #
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0915_mc_s_1_b_0_md_100_nt_100_var_12/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/PIDTags/Test/imgs_ANN.root'.format(ckp),
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/mc_0720_e_pi/Test/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/compare_1_v2'.format(ckp),
    #     ann_threshold_lists=np.power(np.linspace(0, 1, 10000), 6),
    #     bdt_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=1,
    #     source='mc',
    #     n_classes=2,
    #     label_dict=label_dict,
    #     label_list=[0,1]
    # )
    # # cmp.filter_label()
    #
    # cmp.plot_bkg_ratio_compare(x_ll=0.95, x_ul=1, y_ll=1, y_ul=10000000,
    #                            ann_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/ann_detailed_s_1_b_0.csv',
    #                            bdt_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/bdt_detailed_s_1_b_0.csv'
    #                            )
    # cmp.plot_bkg_effi_compare(x_ll=0.9, x_ul=1, y_ll=0.00001, y_ul=1, x_log=False, y_log=True,
    #                           ann_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/ann_detailed_s_1_b_0.csv',
    #                           bdt_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/bdt_detailed_s_1_b_0.csv'
    #                           )
    # cmp.plot_bkg_ratio_imp_factor(
    #     imp_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/ann_info_s_1_b_0.csv',
    #     x_col='ann_effi',
    #     y_col='bkg_ra_imp',
    #     x_ll=0.95,
    #     x_ul=0.99,
    #     y_ll=0,
    #     y_ul=20,
    #     y_scale=1/100,
    #
    # )
    #

    # plot_bkg_ratio_imp_ratio(
    #     imp_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/compare_1_v2/ann_info_s_1_b_0.csv',
    #     x_ll=0.95,
    #     x_ul=0.99,
    #     y_ll=2,
    #     y_ul=22,
    #     y_scale=1,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1/ANA/Fig',
    #     fig_name='bkg_ratio_imp',
    #     source='mc'
    # )

    # main_ana_e_pi(ckp='0915_draft_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1')
    # main_ana_e_pi(ckp='0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_2_1_f_k_7_f_s_2_f_p_3_v1')
    # main_ana_e_pi(ckp='0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_3_1_f_k_7_f_s_2_f_p_3_v1')
    # main_ana_e_pi(
    #     ckp='0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_False_b_1_1_f_k_7_f_s_2_f_p_3_v1')
    # main_sep(
    #     ckp='0915_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_1.0_step_10_st_True_b_1_1_f_k_7_f_s_2_f_p_3_v1',
    #     ep_list=[5, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,110, 120],
    # )
    #
    # main_granularity_cmp()
    #
    # main_ep()

    main_ana_e_pi_16m(ann_ckp='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0120_256_mc_resnet_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_2_l_gamma_0.1_step_10_st_1_fk_3_fs_1_fp_1_v1',
                      bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0210_mc_s_1_b_0_md_100_nt_100_var_12/eval.csv')

    pass
