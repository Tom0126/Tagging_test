#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/1 23:40
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : get_auc_result.py
# @Software: PyCharm

import glob
import numpy as np


def main(file_format, num=10):
    auc_dict = dict()


    for path in glob.glob(file_format):
        auc = np.load(path)

        auc_dict[list(path.split('/'))[-5]] = '{:.2f}'.format(np.mean(auc[1]))  # mean of auc
    auc_dict = sorted(auc_dict.items(), key=lambda x: x[1], reverse=True)

    i=1
    for key, value in auc_dict:
        print(key, value)
        if i>=num:
            break
        i+=1

if __name__ == '__main__':

    main(file_format='/lustre/collider/songsiyuan/TriHiggs/CheckPoint/0331*/ANA/Validation/roc/auroc.npy', num=50)


    pass
