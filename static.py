import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import random as rnd
from scipy.optimize import leastsq, linprog
from scipy import stats

N = 20
y_0, y_1 = 3, 5
xi = np.random.normal(0.0, 1.0, N)
A = np.array
alf_01, alf_02 = -0.3, 0.1
alf_11, alf_12 = 33.3, -33.3
stat = np.zeros(N)
nonstat = np.zeros(N)
stat[0], stat[1] = y_0, y_1
nonstat[0], nonstat[1] = y_0, y_1
t = A(range(N))
for i in range(2, N):
    stat[i] = alf_01*stat[i-1] + alf_02*stat[i-2]
    nonstat[i] = alf_11*nonstat[i-1] + alf_12*nonstat[i-2]

print xi
print stat
print nonstat

plt.plot(t, stat, t, nonstat)
plt.show()

# print xi
