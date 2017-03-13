import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import random as rnd
from scipy.optimize import leastsq, linprog
from scipy import stats
import math

A = np.array
W = lambda k, n: np.e**(-2*math.pi*np.complex(0, 1)*k/n)
a = 3.0
phi = 2
f0 = 1.15
x = np.array([i for i in range(10000)])
y = a*np.sin(2*math.pi*f0*x + phi)


def fft(a):
    n = len(a)
    # print n
    if n == 1:
        return a
    a0 = A([a[i] for i in range(0, n, 2)])[:n/2]
    a1 = A([a[i] for i in range(1, n, 2)])[:n/2]
    # if n%2 == 1:
    #     a1 = np.append(a1, [np.complex(0, 0)])
    a0, a1 = fft(a0), fft(a1)
    w = A([W(k, n) for k in range(len(a0))])
    t1 = a0 + w*a1
    t2 = a0 - w*a1
    res = np.append(t1, t2)
    if len(res) != n:
        return np.append(res, a[n-1:n])
    return res

n0 = 100#2**15
f = 0.5

t = A([i for i in range(n0)])
C = np.cos(2*np.pi*f*t) + np.sin(2*np.pi*(f+0.3)*t)
flec = (2**t)/n0
F1 = np.abs(fft(C))
F2 = np.abs(np.fft.fft(C))
plt.plot(flec, F1)
plt.show()

# t[1:3] = [200, 200]
# print A([W(i, 5) for i in range(5)])
# print W(10)

def rfft(a):
    n = len(a)/2
