#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/25 16:48
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : loader.py
# @Software: PyCharm

import sys

sys.path.append('../..')
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Any
import numpy as np
import torch
import uproot


# based on reading root file


class ReadRoot():

    def __init__(self, file_path, tree_name, start=None, end=None, cut=None, exp=None):
        file = uproot.open(file_path)
        tree = file[tree_name]

        self.tree = tree.arrays(aliases=None,
                                cut=cut,
                                expressions=exp,
                                library="np",
                                entry_start=start,
                                entry_stop=end)

    def readBranch(self, branch):
        return self.tree[branch]

    def build_tensor(self, branch_list, index, input_vector):
        vector = []

        for branch in branch_list:
            vector.append(self.readBranch(branch=branch)[index])

        vector = np.vstack(vector) if input_vector else np.array(vector)

        return vector


class GNNImageSet(Dataset):
    def __init__(self,
                 file_path,
                 max_nodes,
                 padding,
                 exp_dict,
                 tree_name,
                 transform=None) -> None:
        super().__init__()

        self.exp_dict = exp_dict

        self.datasets = ReadRoot(file_path=file_path,
                                 tree_name=tree_name,
                                 exp=exp_dict['vector'] + exp_dict['scalar'] + exp_dict['label']
                                 )

        self.labels = self.datasets.readBranch(branch=exp_dict['label'][0])
        self.labels = self.labels.astype(np.longlong)

        self.transform = transform

        self.max_nodes = max_nodes
        self.padding = padding

    def __getitem__(self, index: Any):

        # point cloud

        points = self.datasets.build_tensor(branch_list=self.exp_dict['vector'], index=index, input_vector=True)

        f_num = points.shape[0]


        # global
        global_vector = self.datasets.build_tensor(branch_list=self.exp_dict['scalar'], index=index, input_vector=False)

        # label
        label = self.labels[index]



        if self.padding:
            if self.max_nodes > points.shape[-1]:

                num_to_padding = self.max_nodes - points.shape[-1]
                choice = np.random.choice(points.shape[-1], num_to_padding, replace=True)
                paddings = points[:, choice]
                # paddings = np.zeros((f_num, num_to_padding))

                points = np.concatenate([points, paddings], axis=-1)

            else:

                choice = np.random.choice(points.shape[-1], self.max_nodes, replace=False)
                points = points[:, choice]

        points = points.astype(np.float32)
        global_vector = global_vector.astype(np.float32)
        # (F, P)

        return points, global_vector, label

    def __len__(self):
        return len(self.labels)


def data_loader_gnn(file_path: str,
                    exp_dict: dict,
                    tree_name: str,
                    batch_size: int = 512,
                    shuffle: bool = False,
                    num_workers: int = 0,
                    max_nodes: int = 32,
                    padding: bool = True,
                    drop_last=True,
                    **kwargs):
    transforms_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_train = GNNImageSet(file_path=file_path,
                                exp_dict=exp_dict,
                                tree_name=tree_name,
                                transform=transforms_train,
                                max_nodes=max_nodes,
                                padding=padding,

                                )

    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=drop_last)

    return loader_train


class MLPImageSet(Dataset):
    def __init__(self,
                 file_path,
                 exp_dict,
                 tree_name,
                 transform=None) -> None:
        super().__init__()

        self.exp_dict = exp_dict

        self.datasets = ReadRoot(file_path=file_path,
                                 tree_name=tree_name,
                                 exp=exp_dict['scalar'] + exp_dict['label']
                                 )

        self.labels = self.datasets.readBranch(branch=exp_dict['label'][0])
        self.labels = self.labels.astype(np.longlong)

        self.transform = transform

    def __getitem__(self, index: Any):
        # global
        global_vector = self.datasets.build_tensor(branch_list=self.exp_dict['scalar'], index=index, input_vector=False)

        # label
        label = self.labels[index]

        global_vector = global_vector.astype(np.float32)

        return global_vector, global_vector, label

    def __len__(self):
        return len(self.labels)


def data_loader_mlp(file_path: str,
                    exp_dict: dict,
                    tree_name: str,
                    batch_size: int = 512,
                    shuffle: bool = False,
                    num_workers: int = 0,
                    drop_last=True,
                    **kwargs):
    transforms_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_train = MLPImageSet(file_path=file_path,
                                exp_dict=exp_dict,
                                tree_name=tree_name,
                                transform=transforms_train,

                                )

    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=drop_last)

    return loader_train


if __name__ == "__main__":
    file_path = '/lustre/collider/wanghaoyu/Ntuples/triHiggs_ML_v2/test/triHiggs_ML.root'

    loader = data_loader_gnn(file_path=file_path,
                             exp_dict={'vector': ['jets_E', 'jets_pt', ],
                                       'scalar': ['circH3', 'circH2', ],
                                       'label': ['isSignal']},

                             tree_name='HHHNtuple',
                             num_workers=0,
                             max_nodes=10,
                             padding=False,
                             batch_size=1)

    for i, (p, g, label) in enumerate(loader):
        print('img:{} g:{} label:{}'.format(p.shape, g.shape, label.shape))
        print(p)
        if i == 1:
            break

    # file = ReadRoot(file_path='/lustre/collider/zhoubaihong/tri-Higgs/ML/Signal_nano.root',
    #                 tree_name='HHHNtuple',
    #                 exp=['jets_E', 'jets_pt', 'circH3', 'circH2', 'isSignal'])
