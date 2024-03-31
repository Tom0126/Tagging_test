#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/27 21:46
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : Evaluate.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
"""
# @file name  : Evaluate.py
# @author     : Siyuan SONG
# @date       : 2023-01-20 12:49:00
# @brief      : CEPC PID
"""

import numpy as np
import pandas as pd

import os
from torch.nn import Softmax
from ANA.acc import plotACC, plotACCbar, plot_purity_threshold, plot_purity_ep
from ANA.distribution import plot_ann_score
from ANA.roc import plotROC, plot_s_b_threshold, plot_s_b_ep, plot_s_b_ratio_threshold, calculate_auc
from ANA.ann_ana import ANN_ANA
from Data.loader import ReadRoot
from torchmetrics.classification import MulticlassROC, MulticlassAUROC, MulticlassAccuracy
# from PID import npyPID, pid_data_loader
from Data.loader import data_loader_gnn

import copy
import torch
import glob


def purity_at_thresholds(model, dataloader, device, num_classes, thresholds_num=100):
    tps = np.zeros((num_classes, thresholds_num))
    nums = np.zeros((num_classes, thresholds_num))
    purities = np.zeros((num_classes, thresholds_num))

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = Softmax(dim=1)(outputs)

            values, predicted = torch.max(outputs, 1)

            for t in range(thresholds_num):
                threshold = (t + 1) / float(thresholds_num)
                cut = values > threshold
                valid_preds = predicted[cut]
                valid_labels = labels[cut]
                for c in range(num_classes):
                    tps[c, t] += ((valid_preds == c) & (valid_labels == c)).cpu().float().sum().item()
                    nums[c, t] += (valid_preds == c).cpu().float().sum().item()

    for c in range(num_classes):
        for t in range(thresholds_num):
            purities[c, t] = tps[c, t] / nums[c, t] if nums[c, t] != 0 else 0

    # print(purities)
    return purities


def totalACC(data_loader, net, device):
    # evaluate
    correct_val = 0.
    total_val = 0.
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):
            # input configuration
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).squeeze().sum().cpu().numpy()
        acc = "{:.2f}".format(100 * correct_val / total_val)
        # print("acc: {}%".format(acc))
        return float(acc)


def ACCParticle(data_loader, net, device, n_classes, threshold=0.9):
    # evaluate
    correct_val = np.zeros(n_classes)
    total_val = np.zeros(n_classes)

    predicts = []
    targets = []
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):
            # input configuration
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            # _, predicted = torch.max(outputs.data, 1)
            #
            # for type in range(n_classes):
            #
            #     total_val[type] += (labels[labels==type]).size(0)
            #     correct_val[type] += (predicted[labels==type] == labels[labels==type]).squeeze().sum().cpu().numpy()

            # acc = 100 * correct_val / total_val

            predicts.append(outputs)
            targets.append(labels)
        targets = torch.cat(targets)
        predicts = torch.cat(predicts)

        mca = MulticlassAccuracy(num_classes=n_classes, average=None, threshold=threshold).to(device)
        acc = 100 * mca(predicts, targets).cpu().numpy()
        # print("acc: {}%".format(acc))
        return acc


def pbDisctuibution(data_loader, net, save_path, device):
    distributions = []
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):

            # input configuration
            inputs = inputs.to(device)

            outputs = net(inputs)
            prbs = Softmax(dim=1)(outputs)
            if j == 0:
                distributions = prbs.cpu().numpy()
            else:
                distributions = np.append(distributions, prbs.cpu().numpy(), axis=0)
        np.save(save_path, distributions)


def getROC(data_loader, net, device, save_path, num_class, ignore_index=None, threshold_num=21):
    preds = torch.tensor([])
    targets = torch.tensor([])
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):

            # input configuration
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            prbs = Softmax(dim=1)(outputs)
            if j == 0:
                preds = prbs
                targets = labels
            else:
                preds = torch.cat((preds, prbs), 0)
                targets = torch.cat((targets, labels), 0)
        metric = MulticlassROC(num_classes=num_class, thresholds=threshold_num, ignore_index=ignore_index).to(device)
        fprs_, tprs_, thresholds_ = metric(preds, targets)
        fprs = []
        tprs = []
        for i, fpr in enumerate(fprs_):
            fprs.append(fpr.cpu().numpy())
            tprs.append(tprs_[i].cpu().numpy())

        np.array(fprs, dtype=object)
        np.array(tprs, dtype=object)
        np.save(save_path.format('fpr'), fprs)
        np.save(save_path.format('tpr'), tprs)

        mc_auroc = MulticlassAUROC(num_classes=num_class, average=None, thresholds=None, ignore_index=ignore_index)
        auroc = mc_auroc(preds, targets)
        np.save(save_path.format('auroc'), auroc.cpu().numpy())


def get_file_name(path):  # get .pth file
    image_files = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.pth':
            return file
    return None


def evaluate(root_dir,
             n_classes,
             net,
             loader_test,
             ana_dir_name='Validation',
             threshold=0,
             threshold_num=21,
             **kwargs
             ):
    # load model

    model_path = os.path.join(root_dir, get_file_name(root_dir))

    ana_dir = os.path.join(root_dir, 'ANA')
    if not os.path.exists(ana_dir):
        os.mkdir(ana_dir)

    save_combin_dir = os.path.join(ana_dir, ana_dir_name)  # all test set combined
    if not os.path.exists(save_combin_dir):
        os.mkdir(save_combin_dir)

    fig_dir = os.path.join(save_combin_dir, 'Fig')
    os.makedirs(fig_dir, exist_ok=True)

    save_combin_path = os.path.join(save_combin_dir, '{}.npy')  # store accuracy
    save_ana_result_path = os.path.join(save_combin_dir, 'eval.csv')

    # TODO ---------------------------check-----------------------------------------------------------------------------

    signals_dict = {
        2: ['bkg', 'signal'],
        3: ['mu+', 'e+', 'pi+'],
        4: ['mu+', 'e+', 'pi+', 'noise']}

    #   distribution

    save_dis_path = os.path.join(fig_dir, 'scores.png')

    # roc

    save_roc_dir = os.path.join(save_combin_dir, 'roc')
    if not os.path.exists(save_roc_dir):
        os.mkdir(save_roc_dir)

    save_roc_path = os.path.join(save_roc_dir, '{}.npy')
    fpr_path = save_roc_path.format('fpr')
    tpr_path = save_roc_path.format('tpr')
    auroc_path = save_roc_path.format('auroc')

    # TODO -------------------------------------------------------------------------------------------------------------

    if torch.cuda.is_available():
        net = net.cuda()
        net.load_state_dict(torch.load(model_path))
        device = 'cuda'
    else:
        device = 'cpu'
        net.load_state_dict(torch.load(model_path, map_location=device))

    correct_val = 0.
    total_val = 0.

    predicts = []
    targets = []

    ana_result_dict = dict()

    with torch.no_grad():
        net.eval()
        for j, (points, vectors, labels) in enumerate(loader_test):
            # input configuration
            points = points.to(device)
            vectors = vectors.to(device)
            labels = labels.to(device)

            outputs = net(points, vectors)
            outputs = Softmax(dim=1)(outputs)

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

            predicts.append(outputs)
            targets.append(labels)

        acc = 100 * correct_val / total_val

        targets = torch.cat(targets)
        predicts = torch.cat(predicts)

        mca = MulticlassAccuracy(num_classes=n_classes, average=None, threshold=threshold).to(device)
        acc_particles = 100 * mca(predicts, targets).cpu().numpy()

        metric = MulticlassROC(num_classes=n_classes, thresholds=threshold_num, ignore_index=None).to(
            device)
        fprs_, tprs_, thresholds_ = metric(predicts, targets)

        fprs = []
        tprs = []

        for i, fpr in enumerate(fprs_):
            fprs.append(fpr.cpu().numpy())
            tprs.append(tprs_[i].cpu().numpy())

        np.array(fprs, dtype=object)
        np.array(tprs, dtype=object)
        np.save(save_roc_path.format('fpr'), fprs)
        np.save(save_roc_path.format('tpr'), tprs)

        mc_auroc = MulticlassAUROC(num_classes=n_classes, average=None, thresholds=None, ignore_index=None)
        auroc = mc_auroc(predicts, targets)
        np.save(save_roc_path.format('auroc'), auroc.cpu().numpy())

    for i, particle in enumerate(signals_dict.get(n_classes)):
        ana_result_dict[particle] = (predicts.cpu().numpy()[:, i]).reshape(-1)

    ana_result_dict['label'] = (targets.cpu().numpy()).reshape(-1)
    ana_result_dict = pd.DataFrame(ana_result_dict)
    ana_result_dict.to_csv(save_ana_result_path)

    plot_ann_score(ana_dict=ana_result_dict, save_path=save_dis_path)

    np.save(save_combin_path.format('combination'), np.array([acc]))

    np.save(save_combin_path.format('acc_particles'), acc_particles)

    save_acc_particle_path = os.path.join(fig_dir, 'acc_particle.png')

    plotACCbar(acc_particles, save_acc_particle_path, threshold)

    # plot roc

    save_roc_fig_path = os.path.join(fig_dir, 'roc.png')
    save_roc_threshold_path = os.path.join(fig_dir, 'threshold.png')
    save_roc_threshold_ratio_path = os.path.join(fig_dir, 'ratio_threshold_ann.png')

    plotROC(fpr_path=fpr_path, tpr_path=tpr_path, auroc_path=auroc_path,
            dim=kwargs.get('dim', 1),
            tag=ana_dir_name,
            save_path=save_roc_fig_path, )

    plot_s_b_threshold(fpr_path=fpr_path,
                       tpr_path=tpr_path,
                       dim=kwargs.get('dim', 1),
                       tag=ana_dir_name,
                       save_path=save_roc_threshold_path,
                       threshold_num=threshold_num,
                       )

    plot_s_b_ratio_threshold(fpr_path=fpr_path,
                             tpr_path=tpr_path,
                             dim=kwargs.get('dim', 1),
                             tag=ana_dir_name,
                             save_path=save_roc_threshold_ratio_path,
                             threshold_num=threshold_num,
                             )


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


def get_ann_info(

        ann_scores,
        raw_labels,
        ann_info_save_dir,
        n_classes,
        ann_signal_label_list,
        effi_points,
        export=True,
        detailed=True
):
    for signal in ann_signal_label_list:

        if signal == 0:
            continue  # TODO temporarily only take 1 as the signal

        ann_threshold_lists = np.sort(ann_scores[:, signal])
        ann_threshold_lists = np.unique(ann_threshold_lists)

        label_include = copy.deepcopy(ann_signal_label_list)
        label_include.remove(signal)

        for b in label_include:
            ann_ana = ANN_ANA(
                ann_scores_path=None,
                ann_scores=ann_scores,
                raw_labels_path=None,
                raw_labels=raw_labels,
                save_dir=ann_info_save_dir,
                ann_threshold_lists=ann_threshold_lists,
                ann_signal_label=signal,
                n_classes=n_classes

            )
            ann_ana.filter_label(label_list=[signal, b])
            ann_ana.export_ann_info(effi_points=effi_points,
                                    export=export,
                                    detailed=detailed)


def ana_info(ana_dir,
             n_classes,
             ann_signal_label_list,
             effi_points,
             export,

             ):
    cols = ['ANN_e', 'ANN_pi']
    ann_scores = pd.read_csv(os.path.join(ana_dir, 'imgs_ANN.csv'))
    raw_labels = ann_scores['particle_label'].values

    get_ann_info(
        ann_scores=ann_scores[cols].values,
        raw_labels=raw_labels,
        ann_info_save_dir=ana_dir,
        n_classes=n_classes,
        ann_signal_label_list=ann_signal_label_list,
        effi_points=effi_points,
        export=export,
        detailed=True
    )


# def ann_eval(ckp_dir,
#              data_dir_format,
#              n_classes,
#              net,
#              pid_data_loader_func,
#              max_nodes,
#              ann_signal_label_list,
#              effi_points,
#              ):
#     ann_scores_path_list = []
#     cols = ['ANN_e', 'ANN_pi']
#     ana_dir = os.path.join(ckp_dir, 'ANA')
#
#     os.makedirs(ana_dir, exist_ok=True)
#
#     for _ in glob.glob(data_dir_format):
#         dataset_type = list(_.split('/'))[-1]
#
#         ana_dir = os.path.join(ckp_dir, 'ANA')
#         os.makedirs(ana_dir, exist_ok=True)
#
#         pid_tag_dir = os.path.join(ana_dir, 'PIDTags')
#         os.makedirs(pid_tag_dir, exist_ok=True)
#
#         pid_tag_dir = os.path.join(pid_tag_dir, dataset_type)
#         os.makedirs(pid_tag_dir, exist_ok=True)
#
#         ana_scores_path = os.path.join(pid_tag_dir, 'imgs_ANN.csv')
#
#         ann_scores_path_list.append(ana_scores_path)
#
#         model_path = os.path.join(ckp_dir, get_file_name(ckp_dir))
#         npyPID(file_path=os.path.join(_, 'imgs.npy'),
#                save_path=ana_scores_path,
#                model_path=model_path,
#                n_classes=n_classes,
#                net=net,
#                pid_data_loader_func=pid_data_loader_func,
#                df=True,
#                labels=np.load(os.path.join(_, 'labels.npy')),
#                cols=cols,
#                max_nodes=max_nodes,
#
#                )
#
#     df_list_ = []
#     for _ in ann_scores_path_list:
#         df_ = pd.read_csv(_)
#         df_list_.append(df_)
#
#     ann_scores = pd.concat(df_list_)
#
#     ann_scores.to_csv(os.path.join(ana_dir, 'imgs_ANN.csv'), index=False)
#
#     raw_labels = ann_scores['particle_label'].values
#
#     get_ann_info(
#         ann_scores=ann_scores[cols].values,
#         raw_labels=raw_labels,
#         ann_info_save_dir=ana_dir,
#         n_classes=n_classes,
#         ann_signal_label_list=ann_signal_label_list,
#         effi_points=effi_points
#     )


if __name__ == '__main__':
    pass
