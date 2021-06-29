# -*- coding: utf-8 -*-
"""
 @File    : crnn.py
 @Time    : 2019/12/2 下午8:21
 @Author  : yizuotian
 @Description    :
"""

from collections import OrderedDict

import torch
from torch import nn
import pickle
import cv2
import numpy as np
import itertools
import os
import requests

class CRNN(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(CRNN, self).__init__(**kwargs)
        self.cnn = nn.Sequential(OrderedDict([
            ('conv_block_1', _ConvBlock(1, 64)),  # [B,64,W,32]
            ('max_pool_1', nn.MaxPool2d(2, 2)),  # [B,64,W/2,16]

            ('conv_block_2', _ConvBlock(64, 128)),  # [B,128,W/2,16]
            ('max_pool_2', nn.MaxPool2d(2, 2)),  # [B,128,W/4,8]

            ('conv_block_3_1', _ConvBlock(128, 256)),  # [B,256,W/4,8]
            ('conv_block_3_2', _ConvBlock(256, 256)),  # [B,256,W/4,8]
            ('max_pool_3', nn.MaxPool2d((2, 2), (1, 2))),  # [B,256,W/4,4]

            ('conv_block_4_1', _ConvBlock(256, 512, bn=True)),  # [B,512,W/4,4]
            ('conv_block_4_2', _ConvBlock(512, 512, bn=True)),  # [B,512,W/4,4]
            ('max_pool_4', nn.MaxPool2d((2, 2), (1, 2))),  # [B,512,W/4,2]

            ('conv_block_5', _ConvBlock(512, 512, kernel_size=2, padding=0))  # [B,512,W/4,1]
        ]))

        self.rnn1 = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.transcript = nn.Linear(512, num_classes)

    def forward(self, x):
        """

        :param x: [B, 1, W, 32]
        :return: [B, W,num_classes]
        """
        x = self.cnn(x)  # [B,512,W/16,1]
        x = torch.squeeze(x, 3)  # [B,512,W]
        x = x.permute([0, 2, 1])  # [B,W,512]
        x, h1 = self.rnn1(x)
        x, h2 = self.rnn2(x, h1)
        x = self.transcript(x)
        return x


class CRNNV(nn.Module):
    """
    垂直版CRNN,不同于水平版下采样4倍，下采样16倍
    """

    def __init__(self, num_classes, **kwargs):
        super(CRNNV, self).__init__(**kwargs)
        self.cnn = nn.Sequential(OrderedDict([
            ('conv_block_1', _ConvBlock(1, 64)),  # [B,64,W,32]
            ('max_pool_1', nn.MaxPool2d(2, 2)),  # [B,64,W/2,16]

            ('conv_block_2', _ConvBlock(64, 128)),  # [B,128,W/2,16]
            ('max_pool_2', nn.MaxPool2d(2, 2)),  # [B,128,W/4,8]

            ('conv_block_3_1', _ConvBlock(128, 256)),  # [B,256,W/4,8]
            ('conv_block_3_2', _ConvBlock(256, 256)),  # [B,256,W/4,8]
            ('max_pool_3', nn.MaxPool2d((1, 2), 2)),  # [B,256,W/8,4]

            ('conv_block_4_1', _ConvBlock(256, 512, bn=True)),  # [B,512,W/8,4]
            ('conv_block_4_2', _ConvBlock(512, 512, bn=True)),  # [B,512,W/8,4]
            ('max_pool_4', nn.MaxPool2d((1, 2), 2)),  # [B,512,W/16,2]

            ('conv_block_5', _ConvBlock(512, 512, kernel_size=2, padding=0))  # [B,512,W/4,1]
        ]))

        self.rnn1 = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.transcript = nn.Linear(512, num_classes)

    def forward(self, x):
        """

        :param x: [B, 1, W, 32]
        :return: [B, W,num_classes]
        """
        x = self.cnn(x)  # [B,512,W/16,1]
        x = torch.squeeze(x, 3)  # [B,512,W]
        x = x.permute([0, 2, 1])  # [B,W,512]
        x, h1 = self.rnn1(x)
        x, h2 = self.rnn2(x, h1)
        x = self.transcript(x)
        return x

class Predict:
    def __init__(self):
        if os.path.isfile('.crnn_label.pk') == False:
            self.download_file('http://download.homescenario.com:8765/crnn/label.pk','.crnn_label.pk')
        if os.path.isfile('.crnn.horizontal.132.pth') == False:
            self.download_file('http://download.homescenario.com:8765/crnn/crnn.horizontal.132.pth','.crnn.horizontal.132.pth')
        alpha = pickle.load(open('.crnn_label.pk','rb'))

        device = torch.device('cpu')
        # 加载权重，水平方向
        h_net = CRNN(num_classes=len(alpha))
        h_net.load_state_dict(torch.load('.crnn.horizontal.132.pth', map_location='cpu')['model'])
        h_net.eval()
        h_net.to(device)
        # 垂直方向
        v_net = CRNNV(num_classes=len(alpha))
        v_net.load_state_dict(torch.load('.crnn.horizontal.132.pth', map_location='cpu')['model'])
        v_net.eval()
        v_net.to(device)
        self.h_net = h_net
        self.v_net = v_net
        self.alpha = alpha
        self.device = device

    def download_file(self, url, name):
        print("download %s" % url)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024000):
                    f.write(chunk)
                    print( ".",end="",flush=True)
            print("\n")


    def predict(self,img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape[:2]
        img = self.pre_process_image(img_gray, h, w)
        text = self.inference(img, h, w)
        text = ''.join(text).replace(' ','')
        return text
    def pre_process_image(self,image, h, w):
        """

        :param image: [H,W]
        :param h: 图像高度
        :param w: 图像宽度
        :return:
        """
        if h != 32 and h < w:
            new_w = int(w * 32 / h)
            image = cv2.resize(image, (new_w, 32))
        if w != 32 and w < h:
            new_h = int(h * 32 / w)
            image = cv2.resize(image, (32, new_h))

        if h < w:
            image = np.array(image).T  # [W,H]
        image = image.astype(np.float32) / 255.
        image -= 0.5
        image /= 0.5
        image = image[np.newaxis, np.newaxis, :, :]  # [B,C,W,H]
        return image


    def inference(self,image, h, w):
        """
        预测图像
        :param image: [H,W]
        :param h: 图像高度
        :param w: 图像宽度
        :return: text
        """
        image = torch.FloatTensor(image)
        image = image.to(self.device)

        if h > w:
            predict = self.v_net(image)[0].detach().cpu().numpy()  # [W,num_classes]
        else:
            predict = self.h_net(image)[0].detach().cpu().numpy()  # [W,num_classes]

        print(predict.shape)
        label = np.argmax(predict[:], axis=1)
        label = [self.alpha[class_id] for class_id in label]
        label = [k for k, g in itertools.groupby(list(label))]
        # label = ''.join(label).replace(' ', '')
        return label


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bn=False):
        super(_ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        if bn:
            self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))


if __name__ == '__main__':
    import torchsummary

    net = CRNN(num_classes=1000)
    torchsummary.summary(net, input_size=(1, 512, 32), batch_size=1)
