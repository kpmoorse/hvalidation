from sysconfig import get_preferred_scheme
import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as skg
import scipy.stats as sps
import scipy.optimize as spo
from tqdm import tqdm
from hval import HVal
import time

# Generate test data
N = 5
xrange = (-1.5,1.5)
noise = 0.01

x1r = np.arange(xrange[0],xrange[1],0.01)
x2r = x1r.copy()
X1r, X2r = np.meshgrid(x1r,x2r)
Xr = np.hstack((X1r.reshape(-1,1),X2r.reshape(-1,1)))
# print(Xr)

x1 = np.random.uniform(xrange[0],xrange[1],N)
x2 = np.random.uniform(xrange[0],xrange[1],N)
X = np.hstack((x1.reshape(-1,1),x2.reshape(-1,1)))
# f_true = lambda x1,x2: (1-(x1**2+x2**2)).reshape(-1,1)
f_true = lambda x1,x2: (0.2-np.abs(1-np.sqrt(x1**2+x2**2))).reshape(-1,1)
y = f_true(x1,x2) + np.random.normal(0,noise,(N,1))

hv = HVal(alpha=noise)
hv.add_data(X,y)

# hv.plot3D(Xr)
# plt.show()

for i in tqdm(range(50)):

    # print(hv.X)
    # print(hv.y)

    xstar = hv.get_max_perr()
    ystar = f_true(xstar[0],xstar[1])+np.random.normal(0,noise)
    hv.add_data(xstar, ystar)

    plt.subplot(121)
    mimg = hv.plot3D(Xr)
    
    plt.subplot(122)
    ax = plt.gca()
    ax.contourf(X1r,X2r,mimg,[-100,0,100])
    ax.set_yticks([])
    ax.set_aspect('equal','box')
    ax.set(xlim=(-1.5,1.5),ylim=(-1.5,1.5))

    circle1 = plt.Circle((0, 0), 1.2, color='w', fill=False, linewidth=2)
    circle2 = plt.Circle((0, 0), 0.8, color='w', fill=False, linewidth=2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # plt.savefig('./demo/hval_demo_{:02d}.png'.format(i))