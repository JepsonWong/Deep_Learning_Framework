#coding:utf-8

import numpy as np

X = [1, 2]
state = [0.0, 0.0]

# 分开定义不同输入部分的权重
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

# 定义用于输出的全连接参数
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

for i in range(len(X)):
    # 循环体中的全连接神经网络 （2 2*2）= 2 （1 2) = 2 2
    before_acrtivation = np.dot(state, w_cell_state) + X[i]*w_cell_input + b_cell
    state = np.tanh(before_acrtivation)

    # 根据当前时刻状态计算输出 (2 2*1) = 1 1
    final_output = np.dot(state, w_output) + b_output

    print "before activation: ", before_acrtivation
    print "state: ", state
    print "output: ", final_output