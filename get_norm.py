#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/16 上午10:41
# @Author  : zhangyunfei
# @File    : get_norm.py
# @Software: PyCharm
import cv2 as cv
import os
import numpy as np
import torch
import torchvision


# 计算数据集的标准差和均值
def init_normalize(data_dir, size):
    img_h, img_w = size[0], size[1]  # 根据自己数据集适当调整，影响不大
    means = [0, 0, 0]
    stdevs = [0, 0, 0]
    img_list = []
    # imgs_path = './data/test'
    imgs_path_list = os.listdir(data_dir)

    num_imgs = 0
    # print(data)
    for pic in imgs_path_list:
        # print(pic)
        num_imgs += 1
        img = cv.imread(os.path.join(data_dir, pic))
        img = cv.resize(img, (img_h, img_w))
        img = img.astype(np.float32) / 255.
        for i in range(3):
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()

    means.reverse()
    stdevs.reverse()
    means = np.asarray(means) / num_imgs
    stdevs = np.asarray(stdevs) / num_imgs
    # print("normMean = {}".format(means))
    # print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
    print(list(means), list(stdevs))
    return list(means), list(stdevs)


if __name__ == '__main__':
    init_normalize('./data/test', [256, 256])
