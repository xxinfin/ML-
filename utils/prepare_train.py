#!/usr/bin/env pytorch
# -*- coding: UTF-8 -*-
"""
@Project     :Follow-Tang-Yudi-machine-learning-algorithm-intensive-code-practice 
@File        :prepare_train.py
@IDE         :PyCharm 
@Author      :张世航
@Date        :2023/3/7 9:24 
@Description :数据预处理
"""
import numpy as np

from utils import generate_polynomials
from utils import generate_sinusoids
from utils import normalize
def prepare_for_train(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """
    对数据进行预处理
    :param data: 原始数据
    :param polynomial_degree: 多项式维度
    :param sinusoid_degree: 正弦维度
    :param normalize_data: 是否进行归一化
    :return: 处理后的数据,特征均值,特征方差
    """
    # 获取样本总数
    num_examples = data.shape[0]
    data_processed = np.copy(data)

    # 预处理
    features_mean = 0
    features_deviation = 0
    data_normalized = data_processed
    if normalize_data:
        data_normalized, features_mean, features_deviation = normalize.normalize(data_processed)
        data_processed = data_normalized

    # 特征变量正弦变换
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids.generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    # 特征变量多项式变换
    if polynomial_degree > 0:
        polynomials = generate_polynomials.generate_polynomials(data_processed, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # 加一列1
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    return data_processed, features_mean, features_deviation
