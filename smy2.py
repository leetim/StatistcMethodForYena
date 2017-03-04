import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import random as rnd
from scipy.optimize import leastsq, linprog
from scipy import stats

a = np.array(map(lambda x: float(x.split(":")[2]), open("yena.txt", "r").read().split()))
p = 6
N = len(a)
x = []

SLV = np.linalg.solve
AP = lambda t, x, alf: sum(alf[k]*x[t-k] for k in range(1, p)) + alf[0]
S = lambda i, k: sum(a[t - i]*a[t-k] for t in range(p, N))
S1 = lambda k: sum(a[t-k] for t in range(p, N))
LSQ = np.linalg.lstsq
m = np.zeros([p, p])
for i in range(1, p):
    for j in range(1, p):
        m[i][j] = S(i, j)
for j in range(1, p):
    m[0][j] = S1(j)
for i in range(1, p):
    m[i][0] = S1(i)
m[0][0] = N - p
b = np.zeros(p)
for i in range(1, p):
    b[i] = S(0, i)
b[0] = S1(0)

x, res, rank, mins = LSQ(m, b)
x1 = SLV(m, b)
print x1
print x
print res
print rank
print mins

a1 = np.zeros(N)
for i in range(p, N):
    a1[i] = AP(i, a, x)


# X = [c - 5 + sum(a[i]*S[n - i-1-tau- zh] for i in range(p)) for tau in range(max_len)]
y = [a[t] for t in range(N)]
for i in range(N*2/3, N):
    y[i] = AP(i, y, x)
T = [i for i in range(1, N+1)]

plt.plot(T, a, T, a1, T, y)
plt.show()

# print m
