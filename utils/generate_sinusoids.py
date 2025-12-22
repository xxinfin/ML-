#!/usr/bin/env pytorch
# -*- coding: UTF-8 -*-
"""
@Project     :Follow-Tang-Yudi-machine-learning-algorithm-intensive-code-practice 
@File        :generate_sinusoids.py
@IDE         :PyCharm 
@Author      :张世航
@Date        :2023/3/8 8:35 
@Description :对参数进行正弦变换
"""
import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """
    变换方式 sin（x)
    :param dataset: 原始数据
    :param sinusoid_degree:变换维度
    :return: 变换后的参数
    """
    num_examples = dataset.shape[0]
    sinusoids = np.empty((num_examples, 0))

    for degree in range(1, sinusoid_degree + 1):
        sinusoid_fatures = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_fatures), axis=1)

    return sinusoids
