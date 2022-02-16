#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/16 上午10:25
# @Author  : zhangyunfei
# @File    : train.py
# @Software: PyCharm
import os
import json
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from captcha_dataset import CaptchaData
from torch.utils.data import DataLoader
import config
import time
import random
import numpy as np
from get_norm import init_normalize
from efficientnet_pytorch import EfficientNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 将数据集划分训练集和验证集
def split_data(files):
    """
    :param files:
    :return:
    """
    random.shuffle(files)
    # 计算比例系数，分割数据训练集和验证集
    ratio = 0.9
    offset = int(len(files) * ratio)
    train_data = files[:offset]
    val_data = files[offset:]
    return train_data, val_data


# 对数据集进行随机打乱
def random_data(files):
    # 设置随机种子，保证每次随机值都一致
    random.seed(2022)
    random.shuffle(files)
    return files


# 计算准确率
def calculat_acc(output, target):
    output, target = output.view(-1, 62), target.view(-1, 62)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc


# 设置随机种子，代码可复现
def seed_it(seed):
    #   random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


# 训练
def train(model, loss_func, optimizer, checkpoints, epochs, lr_scheduler=None):
    print('Train......................')
    # 记录每个epoch的loss和acc
    record = []
    best_acc = 0
    best_epoch = 0
    # 训练过程
    for epoch in range(1, epochs):
        # 设置计时器，计算每个epoch的用时
        start_time = time.time()
        model.train()  # 保证每一个batch都能进入model.train()的模式
        # 记录每个epoch的loss和acc
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        for i, (inputs, labels) in enumerate(train_data):
            # print(i, inputs, labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 预测输出
            outputs = model(inputs)
            # 计算损失
            loss = loss_func(outputs, labels)
            # print(outputs)
            # 因为梯度是累加的，需要清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 优化器
            optimizer.step()
            # 计算准确率
            acc = calculat_acc(outputs, labels)
            train_acc.append(float(acc))
            train_loss.append(float(loss))
        if lr_scheduler:
            lr_scheduler.step()
        # 验证集进行验证
        with torch.no_grad():
            model.eval()
            for i, (inputs, labels) in enumerate(val_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 预测输出
                outputs = model(inputs)
                # 计算损失
                loss = loss_func(outputs, labels)
                # 计算准确率
                acc = calculat_acc(outputs, labels)
                val_acc.append(float(acc))
                val_loss.append(float(loss))

        # 计算每个epoch的训练损失和精度
        train_loss_epoch = torch.mean(torch.Tensor(train_loss))
        train_acc_epoch = torch.mean(torch.Tensor(train_acc))
        # 计算每个epoch的验证集损失和精度
        val_loss_epoch = torch.mean(torch.Tensor(val_loss))
        val_acc_epoch = torch.mean(torch.Tensor(val_acc))
        # 记录训练过程
        record.append(
            [epoch, train_loss_epoch.item(), train_acc_epoch.item(), val_loss_epoch.item(), val_acc_epoch.item()])
        end_time = time.time()
        print(
            'epoch:{} | time:{:.4f} | train_loss:{:.4f} | train_acc:{:.4f} | eval_loss:{:.4f} | val_acc:{:.4f}'.format(
                epoch,
                end_time - start_time,
                train_loss_epoch,
                train_acc_epoch,
                val_loss_epoch,
                val_acc_epoch))

        # 记录验证集上准确率最高的模型
        best_model_path = checkpoints + "/" 'best_model.pth'
        if val_acc_epoch >= best_acc:
            best_acc = val_acc_epoch
            best_epoch = epoch
            torch.save(model, best_model_path)
        print('Best Accuracy for Validation :{:.4f} at epoch {:d}'.format(best_acc, best_epoch))
        # 每迭代50次保存一次模型
        # if epoch % 50 == 0:
        #     model_name = '/epoch_' + str(epoch) + '.pt'
        #     torch.save(model, checkpoints + model_name)
    # 保存最后的模型
    # torch.save(model, checkpoints + '/last.pt')
    # 将记录保存下下来
    record_json = json.dumps(record)
    with open(checkpoints + '/' + 'record.txt', 'w+', encoding='utf8') as ff:
        ff.write(record_json)


if __name__ == '__main__':
    # 设置随机种子
    # seed_it(2022)
    # 分类类别数
    num_classes = config.num_classes
    # batchsize大小
    batch_size = config.batch_size  # config.batch_size
    # 迭代次数epoch
    epochs = config.num_epoch  # config.num_epoch
    # 学习率
    lr = config.lr
    # 模型保存地址
    checkpoints = config.checkpoints
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)
    # 训练接和验证集地址
    train_dir = config.train_dir
    test_dir = config.test_dir
    # 计算均值和标准差
    train_mean, train_std = init_normalize(train_dir, size=[256, 256])
    # 定义图像transform
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 图像放缩
        transforms.RandomRotation((-5, 5)),  # 随机旋转
        # transforms.RandomVerticalFlip(p=0.2),  # 随机旋转
        transforms.ToTensor(),  # 转化成张量
        transforms.Normalize(
            mean=train_mean,
            std=train_std
        )
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 图像放缩
        transforms.ToTensor(),  # 转化成张量
        transforms.Normalize(
            mean=train_mean,
            std=train_std
        )
    ])
    # 装载训练数据
    files = os.listdir(train_dir)
    img_paths = []
    for img in files:
        img_path = os.path.join(train_dir, img)
        img_paths.append(img_path)
    # 将训练数据拆分训练集和验证集
    train_paths, val_paths = split_data(img_paths)
    # 加载训练数据集，转化成标准格式
    train_dataset = CaptchaData(train_paths, transform=train_transform)
    # 加载数据
    train_data = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
    # 加载验证集，转化成标准格式
    val_dataset = CaptchaData(val_paths, transform=val_transform)
    val_data = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    print('训练集数量：%s   验证集数量：%s' % (train_dataset.__len__(), val_dataset.__len__()))

    # 使用框架封装好的模型，使用预训练模型resnet34
    # model = models.resnet34(pretrained=True)
    # # 使用预训练模型需要修改fc层
    # num_fcs = model.fc.in_features
    # # print(num_fcs)
    # model.fc = nn.Sequential(
    #     nn.Linear(num_fcs, num_classes)
    # )
    # 使用efficientnet网络模型
    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=248)
    # print(model)
    # GPU是否可用，如果可用，则使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 如果是多个gpu，数据并行训练
    # device_ids = [0, 1, 3, 4, 5]
    # model = torch.nn.DataParallel(model, device_ids=device_ids)

    # 定义损失函数
    loss_func = nn.MultiLabelSoftMarginLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 训练
    train(model, loss_func, optimizer, checkpoints, epochs)
