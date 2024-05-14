#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/7 10:26
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : read_root.py
# @Software: PyCharm
import os

import pandas as pd
import uproot


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

    def build_csv(self, branch_list, save_path):
        csv_file = dict()

        for branch in branch_list:
            csv_file[branch] = self.readBranch(branch=branch)

        csv_file = pd.DataFrame(csv_file)

        csv_file.to_csv(save_path)


if __name__ == '__main__':

    tree= ReadRoot(file_path='/lustre/collider/wanghaoyu/Ntuples/triHiggs_ML_v5/validation/triHiggs_ML.root',
                   exp=['jets_pt'],
                   start=0,
                   end=10,
                   tree_name='HHHNtuple')

    eta=tree.readBranch('jets_pt')

    for _ in eta:
        print(
            _
        )

    pass
