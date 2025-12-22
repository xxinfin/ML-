#!/usr/bin/env pytorch
# -*- coding: UTF-8 -*-
"""
@Project     :Follow-Tang-Yudi-machine-learning-algorithm-intensive-code-practice 
@File        :generate_polynomials.py
@IDE         :PyCharm 
@Author      :张世航
@Date        :2023/3/8 8:34 
@Description :进行多项式变换增加变量复杂度
"""
import numpy as np
from utils import normalize


def generate_polynomials(dataset, polynomials_degree, normalize_data=False):
    """
    变换方法：x1, x2, x1^2, x2^2, x1 * x2, x1 * x2^2, etc.
    :param dataset:原始数据
    :param polynomials_degree:多项式的维度
    :param normalize_data:是否归一化
    :return:生成的多项式参数
    """
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    if num_examples_1 != num_examples_2:
        raise ValueError("can not generate polynomials for two sets with different number")
    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError("can not generate polynomials for two sets with no colums")
    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    num_features = num_features_1 if num_features_1 < num_examples_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    polynomials = np.empty((num_examples_1, 0))
    for i in range(1, polynomials_degree + 1):
        for j in range(i + 1):
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    if normalize_data:
        polynomials = normalize.normalize(polynomials)[0]

    return polynomials
