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
speed = 2.3/(10**5) #Iron
h = 0.02
A = np.array
T = np.transpose
SOLVE = np.linalg.solve
n = 100
K = lambda i, j: (n-2)*(i-1) + j - 1
lim_x = A([30, 50])
lim_y = A([39, 50])
F = lambda i, j: (0.0, -(h*h)/(speed)*20.0)[((lim_x[0] <= j) & (lim_x[1] >= j)) & ((lim_y[0] <= i) & (lim_y[1] >= i))]
F_T = lambda i, j: ((0 < j) & (n-1 > j)) & ((0 < i) & (n-1 > i))

u = np.zeros([n, n])
f = np.zeros([n, n])
# f =
f = A([[F(i, j) for j in range(n)] for i in range(n)])
f[4][1] = -(h*h)/(speed)*20.0
f[1][4] = -(h*h)/(speed)*20.0
print "f:"
print f
a = np.zeros([(n-2)**2, (n-2)**2])
b = np.zeros((n-2)**2)

for i in range(1, n-1):
    for j in range(1, n-1):
        print K(i, j)
        if F_T(i, j):#K(i, j) > 0 & K(i, j) < (n-1)**2-1:
            a[K(i, j)][K(i, j)] = -4
        if F_T(i, j-1):#K(i, j-1) > 0 & K(i, j-1) < (n-1)**2-1:
            a[K(i, j)][K(i, j-1)] = 1
        if F_T(i, j+1):#K(i, j+1) > 0 & K(i, j+1) < (n-1)**2-1:
            a[K(i, j)][K(i, j+1)] = 1
        if F_T(i-1, j):#K(i-1, j) > 0 & K(i-1, j) < (n-1)**2-1:
            a[K(i, j)][K(i-1, j)] = 1
        if F_T(i+1, j):#K(i+1, j) > 0 & K(i+1, j) < (n-1)**2-1:
            a[K(i, j)][K(i+1, j)] = 1
        b[K(i, j)] = f[i][j]


print a
x = SOLVE(a, b)
for i in range(1, n-1):
    for j in range(1, n-1):
        u[i][j] = x[K(i, j)]
print "u:"
print u
# print f[40:50, 30:45]

# make these smaller to increase the resolution
# dx, dy = 0.05, 0.05

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[slice(0, h*n, h),
                slice(0, h*n, h)]
print len(y[0])
z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = u[:-1, :-1]
# print len(z), len(z[0])
levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())


# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
print levels

fig, (ax0, ax1) = plt.subplots(nrows=2)

im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax0)
ax0.set_title('pcolormesh with levels')


# contours are *point* based plots, so convert our bound into point
# centers
cf = ax1.contourf(x[:-1, :-1] + h/2.,
                  y[:-1, :-1] + h/2., z, levels=levels,
                  cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title('contourf with levels')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()

plt.show()
