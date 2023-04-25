from sysconfig import get_preferred_scheme
import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as skg
import scipy.stats as sps
import scipy.optimize as spo
from tqdm import tqdm
import hval
import time
import enz

t = np.arange(0,20,0.001)
th0_list = np.arange(-np.pi/4, np.pi/4, 0.1).reshape(-1,1)
init_list = np.hstack((th0_list, np.zeros_like(th0_list)))

# b=(0.1,0.3,0.05)
# nlc = sim_pendulum((np.pi/3,0),t,nl_compound,b1=b[0],b2=b[1],b3=b[2])
# nls = sim_pendulum((np.pi/3,0),t,nl_simple,b1=b[0],b2=b[1],b3=b[2])
# ls = sim_pendulum((np.pi/3,0),t,lin_simple,b1=b[0],b2=b[1],b3=b[2])

# # g = []
# # for i in init_list:
# #     g.append(compare(i, t, nl_compound, nl_simple, sperf, thresh=50))
# # g = np.array(g)

# plt.plot(t,nlc[:,0],t,nls[:,0],t,ls[:,0])
# plt.ylim(-1.25,1.25)
# plt.legend(["NL, compound", "NL, simple","Linear, simple"])
# plt.show()

t = np.linspace(0, 1500, 100)
def compare(phi,t,mi,mj,g):

    di = mi(phi,t)
    dj = mj(phi,t)
    # print(di,dj)

    gg = g(di,dj)
    # print(gg)
    return gg

x1range = (0,100)
x2range = (0,100)

Ni = 5
init_x1 = np.random.uniform(x1range[0],x1range[1],(Ni,1))
init_x2 = np.random.uniform(x2range[0],x2range[1],(Ni,1))
init_X = np.hstack((init_x1,init_x2))
init_X = init_x1.reshape((-1,1))

init_y = [compare(init_X[i,:],t,enz.sim_m1,enz.sim_m2,enz.g) for i in range(Ni)]
init_y = np.array(init_y).reshape(-1,1)

init_y2 = [compare(init_X[i,:],t,enz.sim_m1,enz.sim_m3,enz.g) for i in range(Ni)]
init_y2 = np.array(init_y2).reshape(-1,1)

hv = hval.HVal()
hv.add_data(init_X,init_y)
hv2 = hval.HVal()
hv2.add_data(init_X,init_y2)

x1r = np.arange(x1range[0],x1range[1],0.1)
x2r = np.arange(x2range[0],x2range[1],1)
X1r, X2r = np.meshgrid(x1r,x2r)
Xr = np.hstack((X1r.reshape(-1,1),X2r.reshape(-1,1)))
# mimg = hv.plot3D(Xr)
# plt.gca().set(xlim=(-np.pi/2,np.pi/2),ylim=(-np.pi/2,np.pi/2))
# plt.show()

# Sample against model 1 (nonlinear)
for i in tqdm(range(20)):

    # print(hv.X)
    # print(hv.y)

    xstar = hv.get_max_perr()
    # print(hv.X)
    # print(xstar)
    ystar = compare(xstar,t,enz.sim_m1,enz.sim_m2,enz.g)
    ystar2 = compare(xstar,t,enz.sim_m1,enz.sim_m3,enz.g)
    # print(ystar)
    hv.add_data(xstar.reshape(-1,2), ystar)
    hv2.add_data(xstar.reshape(-1,2), ystar2)

    hv.plot2D()

    # plt.subplot(121)
    plt.cla()
    mimg2 = hv2.plot3D(Xr)
    mimg = hv.plot3D(Xr)
    # plt.text(-1.5,-1.5,"mdl1",fontsize="xx-large")
    # plt.clf()
    # # plt.colorbar()

    # plt.subplot(122)
    # plt.cla()
    # ax = plt.gca()
    # ax.imshow((mimg>0).astype(int)+(mimg2>0).astype(int), extent=[0,100,0,1000],origin='lower')
    # # ax.contourf(X1r,X2r,mimg,[-100,0,100])
    # ax.set_yticks([])
    # # ax.set_aspect('equal','box')
    # ax.set(xlim=x1range,ylim=x2range)

    plt.show()
    # plt.savefig('./demo/enz_demo_a{:02d}.png'.format(i))

    print(hv.y)
    print(hv2.y)

# # Sample against model 2 (linear)
# for i in tqdm(range(10)):

#     # print(hv.X)
#     # print(hv.y)

#     xstar = hv2.get_max_perr()
#     # print(hv.X)
#     # print(xstar)
#     ystar2 = compare(xstar,t,enz.sim_m1,enz.sim_m3,enz.g)
#     # print(ystar)
#     hv2.add_data(xstar.reshape(-1,2), ystar2)

#     plt.subplot(121)
#     plt.cla()
#     mimg = hv.plot3D(Xr)
#     mimg2 = hv2.plot3D(Xr)
#     # plt.text(-1.5,-1.5,"mdl2",fontsize="xx-large")

#     plt.subplot(122)
#     plt.cla()
#     ax = plt.gca()
#     ax.imshow((mimg>0).astype(int)+(mimg2>0).astype(int), extent=[-np.pi/2,np.pi/2,-np.pi/2,np.pi/2],origin='lower')
#     # ax.contourf(X1r,X2r,mimg,[-100,0,100])
#     ax.set_yticks([])
#     ax.set_aspect('equal','box')
#     ax.set(xlim=(-np.pi/2,np.pi/2),ylim=(-np.pi/2,np.pi/2))

#     # plt.show()
#     plt.savefig('./demo/enz_demo_b{:02d}.png'.format(i))