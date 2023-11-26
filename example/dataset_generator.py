"""
A collection of functions to generate datasets for testing and training.
"""


import numpy as np
import random
import math


def arithmetic_ops(points):
    X, Y = [], []
    classes = [0, 1, 2, 3]
    for class_number in classes:
        for _ in range(int(points/4)):
            num1 = random.random()*10
            num2 = random.random()*10
            if class_number == 0:
                op = num1 + num2
            elif class_number == 1:
                op = num1 - num2
            elif class_number == 2:
                op = num1 * num2
            elif class_number == 3:
                op = num1 / num2
            X.append((num1, num2, op))
            Y.append(class_number)
    return np.array(X), np.array(Y)


def sine_wave_data(points, classes):
    X, Y = [], []
    for class_number in range(1, classes+1):
        for _ in range(int(points/3)):
            x = random.random()*10
            y = math.sin(class_number * x)
            X.append((x, y))
            Y.append(class_number - 1)

    return np.array(X), np.array(Y)


def pn_data(points):
    X = np.zeros((points, 2))
    y = np.zeros(points, dtype='uint8')
    for ix in range(points):
        n1 = random.randint(-10, 10)
        n2 = random.randint(-10, 10)
        if n1*n2 < 0:
            X[ix] = np.c_[n1, n2]
            y[ix] = 0
        else:
            X[ix] = np.c_[n1, n2]
            y[ix] = 1
    return X, y


def prime_data(points):
    X, Y = [], []
    X.extend([[2], [3], [4], [5], [6], [7], [8], [9]])
    Y.extend([1, 1, 0, 1, 0, 1, 0, 0])
    for x in range(10, points):
        flag = True
        for i in range(2, int(x/2)+1):
            if x % i == 0:
                flag = False
            else:
                pass
        if flag:
            Y.append(1)
        else:
            Y.append(0)
        X.append([x])
    return np.array(X), np.array(Y)


def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4,
                        points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


def vertical_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        X[ix] = np.c_[np.random.randn(
            samples)*.1 + (class_number)/3, np.random.randn(samples)*.1 + 0.5]
        y[ix] = class_number
    return X, y


def linear_data(hm, variance, step=2, correlation=True):
    val = 1
    ys = []
    for _ in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation or correlation == 'pos':
            val += step
        elif correlation or correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
