#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/16 上午10:28
# @Author  : zhangyunfei
# @File    : config.py
# @Software: PyCharm
"""
    配置文件
"""
# 图像大小
IMAGE_HEIGHT = 40
IMAGE_WIDTH = 100
# 图像分类的类别
num_classes = 248
# 训练batchsize大小
batch_size = 32
# 训练epoch
num_epoch = 200
# 学习率
lr = 0.001
# 训练过程保存模型地址
checkpoints = 'checkpoints'

# 训练集和验证集
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'
