#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/16 上午10:27
# @Author  : zhangyunfei
# @File    : predict.py
# @Software: PyCharm
import os
import cv2 as cv
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from get_norm import init_normalize
import csv

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97 + 26)]
source += [chr(i) for i in range(65, 65 + 26)]
alphabet = ''.join(source)


def predict():
    model_path = './checkpoints/best_model.pth'
    test_dir = './data/test'
    test_mean, test_std = init_normalize(test_dir, size=[256, 256])
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=test_mean,
            std=test_std
        )
    ])
    print(torch.cuda.is_available())
    model = torch.load(model_path)
    if torch.cuda.is_available():
        model = model.cuda()

    images = os.listdir(test_dir)
    images.sort(key=lambda x: int(x[:-4]))
    res = []
    for img in images:
        img_path = os.path.join(test_dir, img)
        image_read = Image.open(img_path)
        gray = image_read.convert('RGB')
        gray = transform(gray)
        image = gray.view(1, 3, 256, 256).cuda()
        output = model(image)
        output = output.view(-1, 62)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        output = output.view(-1, 4)[0]
        pred = ''.join([alphabet[i] for i in output.cpu().numpy()])
        # print([alphabet[i] for i in output.cpu().numpy()])
        print(img, pred)
        res.append({'num': int(img[:-4]), 'tag': pred})

    header = ['num', 'tag']
    os.makedirs('sub', exist_ok=True)
    with open('sub/submit_021601.csv', 'w', encoding='utf_8_sig') as f:
        f_csv = csv.DictWriter(f, header)
        f_csv.writeheader()
        f_csv.writerows(res)


if __name__ == '__main__':
    predict()
