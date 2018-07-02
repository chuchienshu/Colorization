# -*- coding: utf-8 -*-
import numpy as np
from os.path import *
from scipy.misc import imread
gray_sub = imread('/home/chuchienshu/Downloads/dataset/DAVIS_test/boxing/00002.jpg')
def read_gen(file_name):
    ext = splitext(file_name)[-1].lower()
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(file_name)
        if len(im.shape) == 2:
            return gray_sub
        # if im.shape[2] > 3:
        #     return im[:,:,:3]
        # else:
        return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    return []