#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-02-17 11:03:35
#   Description :
#
#================================================================

import core.common as common
import tensorflow as tf


def darknet53(input_data, trainable):

    with tf.variable_scope('darknet'):
        #filters_shape=(3, 3,  3,  32),前2个3代表卷积核大小，第三个3代表输入图像的通道数数，此处为RGB，32是输出的通道数
        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        #conv0 416*416*3 ---- 416*416*32
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)
        # conv1 416*416*32 ---- 208*208*64
        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))
        #residual 0 208*208*64---208*208*64
        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)
        # conv4 208*208*64---104*104*128
        for i in range(2):
            input_data = common.residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))
        # residual 1 104*104*128---104*104*128
        # residual 2 104*104*128---104*104*128
        route_1 = input_data #4倍下采样的输出，104*104*128
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)
        # conv9 104*104*128---52*52*256
        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))
        # residual 3 52*52*256---52*52*128
        # residual 4 52*52*128---52*52*256
        # residual 5 52*52*256---52*52*128
        # residual 6 52*52*128---52*52*256
        # residual 7 52*52*256---52*52*128
        # residual 8 52*52*128---52*52*256
        # residual 9 52*52*256---52*52*128
        # residual 10 52*52*128---52*52*256
        route_2 = input_data #用作后面的concat操作，是8倍下采样的输出52*52*256
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)
        # conv26 52*52*256---26*26*512
        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))
        # residual 11 26*26*512---26*26*256
        # residual 12 26*26*256---26*26*512
        # residual 13 26*26*512---26*26*256
        # residual 14 26*26*256---26*26*512
        # residual 15 26*26*512---26*26*256
        # residual 16 26*26*256---26*26*512
        # residual 17 26*26*512---26*26*256
        # residual 18 26*26*256---26*26*512
        route_3 = input_data #16倍下采样的输出26*26*512
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)
        # conv43 26*26*512---13*13*1024
        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))
        # residual 19 13*13*1024---13*13*512
        # residual 20 13*13*512---13*13*1024
        return route_1, route_2, route_3, input_data




