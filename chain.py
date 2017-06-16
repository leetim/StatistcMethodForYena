import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import random as rnd
from scipy.optimize import leastsq, linprog
from scipy import stats
from copy import copy
import math

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

A = np.array
T = np.transpose

l = 1.0
n = 20
p1 = (0.0, 6.0)
p2 = (4.0, 0.0)
Y = A([p1[1] + (p2[1] - p1[1])*i/n for i in range(n+1)])
X = A([p1[0] + (p2[0] - p1[0])*i/n for i in range(n+1)])

def Gi(X, Y, i):
    return np.sqrt((X[i+1] - X[i])**2 + (Y[i+1] - Y[i])**2)
def G(X, Y):
    return sum(Gi(X, Y, i) for i in range(n))
def Cyi(X, Y, i):
    return (Y[i] + Y[i+1])/2
def Cy(X, Y):
    Gt = G(X, Y)
    return sum(Gi(X, Y, i)*Cyi(X, Y, i) for i in range(n))/Gt
def PHIi(X, Y, i):
    return (Gi(X, Y, i)**2 - l**2)**2
def PHI(X, Y):
    return sum(PHIi(X, Y, i) for i in range(n))

def Func(X, Y):
    return - Cy(X, Y) - PHI(X, Y)
    # return - PHI(X, Y)
print Func(X, Y)
def dFuncHX(X, Y, j, h):
    d = A([ h if i == j else 0.0 for i in range(n+1)])
    nX = X + d
    return (Func(nX, Y) - Func(X, Y))/h
def dFuncHY(X, Y, j, h):
    d = A([ h if i == j else 0.0 for i in range(n+1)])
    nY = Y + d
    return (Func(X, nY) - Func(X, Y))/h

def get_next(X, Y):
    h = 0.001
    grad_x = np.zeros(n+1)
    grad_y = np.zeros(n+1)
    for i in range(1, n):
        grad_x[i] = dFuncHX(X, Y, i, h)
        grad_y[i] = dFuncHY(X, Y, i, h)
    eps = 1.0
    tX, tY = copy(X), copy(Y)
    while eps > h:
        nX = tX + eps*grad_x
        nY = tY + eps*grad_y
        if Func(nX, nY) > Func(tX, tY):
            tX, tY = nX, nY
        else:
            eps /= 2
    return tX, tY

while True:
    nX, nY = get_next(X, Y)
    print PHI(nX, nY)
    print Func(nX, nY)
    if np.abs(Func(X, Y) - Func(nX, nY)) < 0.000001:
        break
    X, Y = nX, nY
print A([PHIi(X, Y, i) for i in range(n)])
plt.plot(X, Y)
plt.show()
