import ctypes
from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer


def str2arg(path):
    path_string = path.encode('utf-8')
    arg = c_char_p(path_string)
    return arg


def get_temper_info(ip='192.168.1.65', username='admin', password='Admin12345'):
    f = ctypes.CDLL('./libtemperature.so').getTemperature
    f.argtypes = [c_char_p, c_char_p, c_char_p]  # 定义输入类型
    f.restype = ndpointer(dtype=ctypes.c_float, shape=(288 * 384))  # 定义输出类型

    res = f(str2arg(ip), str2arg(username), str2arg(password))
    res1 = res.reshape(288, 384)  # 得到的温度图是个numpy二维数组
    xmin = 0
    xmax = 384
    ymin = 0
    ymax = 288

    part = res1[ymin:ymax, xmin:xmax]

    min_temper = np.min(part)
    max_temper = np.max(part)
    aver_temp = np.mean(part)

    return min_temper, max_temper, aver_temp


if __name__ == '__main__':
    print(get_temper_info())
