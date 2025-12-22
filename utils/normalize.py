#!/usr/bin/env pytorch
# -*- coding: UTF-8 -*-
"""
@Project     :Follow-Tang-Yudi-machine-learning-algorithm-intensive-code-practice 
@File        :normalize.py
@IDE         :PyCharm 
@Author      :张世航
@Date        :2023/3/7 16:02 
@Description :归一化数据
"""
import numpy as np


def normalize(features):
    """
    特征归一化
    :param features: 传入特征
    :return: 归一化后的特征,特征均值,特征标准差
    """
    features_normalized = np.copy(features).astype(float)
    # 计算均值
    features_mean = np.mean(features, 0)
    # 计算标准差
    features_deviation = np.std(features, 0)
    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean
    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation
    return features_normalized, features_mean, features_deviation
