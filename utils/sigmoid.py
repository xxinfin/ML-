#!/usr/bin/env pytorch
# -*- coding: UTF-8 -*-
"""
@Project     :Follow-Tang-Yudi-machine-learning-algorithm-intensive-code-practice 
@File        :sigmoid.py
@IDE         :PyCharm 
@Author      :张世航
@Date        :2023/3/15 15:57 
@Description :sigmoid函数的定义
"""
import math

import matplotlib.pyplot as plt
import numpy as np


def sigmoid_function(z):
    """
    sigmoid函数，把参数归一到0-1中
    :param z:输入是x
    :return: 输出的f(x)
    """
    fz = []
    for num in z:
        fz.append(1 / (1 + math.exp(-num)))
    return fz


def sigmoid(matrix):
    """
    矩阵的sigmoid方法 
    :param matrix: 输入的矩阵
    :return: sigmoid后的矩阵
    """
    return 1 / (1 + np.exp(-matrix))


def sigmoid_gradient(matrix):
    """
    sigmoid函数的导数
    :param matrix: 输入的矩阵
    :return: sigmoid导数矩阵
    """
    return np.dot(sigmoid(matrix).T, 1 - sigmoid(matrix))[0][0]


if __name__ == '__main__':
    z = np.arange(-10, 10, 0.01)
    fz = sigmoid_function(z)
    plt.title("sigmoid function")
    plt.xlabel("z")
    plt.ylabel("f(z)")
    plt.plot(z, fz)
    plt.show()
