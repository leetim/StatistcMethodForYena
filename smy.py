import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import random as rnd
from scipy.optimize import leastsq, linprog
from scipy import stats

S = np.array(map(lambda x: float(x.split(":")[2]), open("yena.txt", "r").read().split()))
p = 5
n = len(S)

def auto_regr(k , S):
    return sum(a[i]*S[k - p + i] for i in range(p))

R = [sum([S[j]*S[j + i] for j in range(n - i)])/n for i in range(p+1)]
print R

a = np.zeros(p+1)
K = np.zeros(p+1)
E = np.zeros(p+1)
E[0] = R[0]
for l in range(1, p+1):
    K[l] = (sum(a[i]*R[l - i] for i in range(l)) - R[l])/E[l-1]
    E[l] = E[l-1]*(1 - K[l]**2)
    for i in range(1, l):
        a[i] = a[i] + K[l]*a[i-1]
    a[l] = -K[l]
print a
# print S
# tau = 7
c = -sum(sum(a[i]*S[n - i-1-tau] for i in range(p)) for tau in range(n - p))/(n - p) + sum(S[p+i] for i in range(n - p))/(n-p)
for tau in range(30):
    print c + sum(a[i]*S[n - i-1-tau] for i in range(p))
    print S[n - tau-1]
zh = 30
max_len = 15
X = [c - 5 + sum(a[i]*S[n - i-1-tau- zh] for i in range(p)) for tau in range(max_len)]
Y = [S[n - tau-1 - zh] for tau in range(max_len)]
Z = np.zeros(max_len)
for i in range(p):
    Z[i] = S[n - i - 1 - zh]
for i in range(p, max_len):
    Z[i] = c - 5 + sum(a[j]*Z[i - j - 1] for j in range(p))
t = [i for i in range(max_len)]
plt.plot(t, X, t, Y, t, Z)
print a
plt.show()
