#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 12:44
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : check_label.py
# @Software: PyCharm



import numpy as np
from collections import Counter

def check_label(file_path):
    labels=np.load(file_path)
    count=Counter(labels)
    print(count)


if __name__ == '__main__':

    file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/mc_0720_e_pi_block_1_1/TV/labels.npy'
    check_label(file_path)
    pass
