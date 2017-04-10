import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import random as rnd
from scipy.optimize import leastsq, linprog
from scipy import stats
import math

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


speed = 1.27/(10**4) #Gold
# speed = 2.3/(10**5) #Iron
h = 0.05
A = np.array
T = np.transpose
SOLVE = np.linalg.solve
n = 100
K = lambda i, j: (n-2)*(i-1) + j - 1
lim_x = A([10, 90])
lim_y = A([39, 39])
F = lambda i, j: (0.0, -(h*h)/(speed)*20.0)[((lim_x[0] <= j) & (lim_x[1] >= j)) & ((lim_y[0] <= i) & (lim_y[1] >= i))]
F_T = lambda i, j: ((0 < j) & (n-1 > j)) & ((0 < i) & (n-1 > i))

u = np.zeros([n, n])
f = np.zeros([n, n])
# f =
f = A([[F(i, j) for j in range(n)] for i in range(n)])
f[20][20] = -(h*h)/(speed)*20.0
f[20][21] = -(h*h)/(speed)*20.0
# print "f:"
# print f
a = np.zeros([(n-2)**2, (n-2)**2])
b = np.zeros((n-2)**2)

for i in range(1, n-1):
    for j in range(1, n-1):
        # print K(i, j)
        if F_T(i, j):
            a[K(i, j)][K(i, j)] = -4
        if F_T(i, j-1):
            a[K(i, j)][K(i, j-1)] = 1
        if F_T(i, j+1):
            a[K(i, j)][K(i, j+1)] = 1
        if F_T(i-1, j):
            a[K(i, j)][K(i-1, j)] = 1
        if F_T(i+1, j):
            a[K(i, j)][K(i+1, j)] = 1
        b[K(i, j)] = f[i][j]

# print a
x = SOLVE(a, b)
for i in range(1, n-1):
    for j in range(1, n-1):
        u[i][j] = x[K(i, j)]
y, x = np.mgrid[slice(0, h*n, h),
                slice(0, h*n, h)]

z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
z = u[:-1, :-1]

levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

cmap = plt.get_cmap('OrRd')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# print levels

fig, ax1 = plt.subplots(nrows=1)

cf = ax1.contourf(x[:-1, :-1] + h/2.,
                  y[:-1, :-1] + h/2., z, levels=levels,
                  cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title('temperature')

fig.tight_layout()
plt.show()
