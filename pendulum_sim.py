from curses import nl
import numpy as np
import matplotlib.pyplot as plt
import hval
from tqdm import tqdm

def sim_pendulum(initial_state, t, diffeq, g=9.8, l=1, b1=0.1, b2=0.3, b3=0.05):

    dt = np.diff(t)[0]

    state = [initial_state]

    for k in range(len(t)-1):
        th, thd = state[-1]
        thdd = diffeq(th, thd, (g,l,b1,b2,b3))

        state.append([
            th+thd*dt,
            thd+thdd*dt
        ])

    return np.array(state)

# Simulate a nonlinear pendulum with compound drag model
def nl_compound(th, thd, params):

    # b1 corresponds to air resistance ~v**2
    # b2, b3 correspond to hinge drag ~v,sgn(v)
    g,l,b1,b2,b3 = params
    return -g*l*np.sin(th) - (b1*np.sign(thd)*thd**2 + b2*thd + b3*np.sign(thd))

# Simulate a nonlinear pendulum with simplified drag model
def nl_simple(th, thd, params):

    g,l,b1,b2,b3 = params
    return -g*l*np.sin(th) - (b2*thd + b3*np.sign(thd)) #(b1*np.sign(thd)*thd**2)#

# Simulate a linearized pendulum with simplified drag model
def lin_simple(th, thd, params):

    g,l,b1,b2,b3 = params
    return -g*l*th - (b2*thd + b3*np.sign(thd))

def pperf(state1, state2, thresh):

    # Calculate "periodic distance" between position vectors
    F1 = np.abs(np.fft.fft(state1[:,0]))
    F2 = np.abs(np.fft.fft(state2[:,0]))
    pnorm = np.linalg.norm(F1-F2)
    
    return thresh-pnorm

def sperf(state1, state2, thresh=0):

    norm = np.linalg.norm(state1-state2)
    return thresh - norm

def compare(initial_state, t, model1, model2, metric, thresh=25):

    state1 = sim_pendulum(initial_state, t, model1)
    state2 = sim_pendulum(initial_state, t, model2)

    return metric(state1, state2, thresh)

if __name__ == '__main__':

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

    Ni = 5
    init_th = np.random.uniform(-np.pi/2,np.pi/2,(Ni,1))
    init_thd = np.random.uniform(-np.pi/2,np.pi/2,(Ni,1))
    init_X = np.hstack((init_th,init_thd))

    init_y = [compare(init_X[i,:],t,nl_compound,nl_simple,sperf) for i in range(Ni)]
    init_y = np.array(init_y).reshape(-1,1)

    init_y2 = [compare(init_X[i,:],t,nl_compound,lin_simple,sperf) for i in range(Ni)]
    init_y2 = np.array(init_y2).reshape(-1,1)

    hv = hval.HVal()
    hv.add_data(init_X,init_y)
    hv2 = hval.HVal()
    hv2.add_data(init_X,init_y2)

    xrange = (-np.pi/2,np.pi/2)
    x1r = np.arange(xrange[0],xrange[1],0.01)
    x2r = x1r.copy()
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
        ystar = compare(xstar,t,nl_compound,nl_simple,sperf)
        ystar2 = compare(xstar,t,nl_compound,lin_simple,sperf)
        # print(ystar)
        hv.add_data(xstar.reshape(-1,2), ystar)
        hv2.add_data(xstar.reshape(-1,2), ystar2)
    
        plt.subplot(121)
        plt.cla()
        mimg2 = hv2.plot3D(Xr)
        mimg = hv.plot3D(Xr)
        plt.text(-1.5,-1.5,"mdl1",fontsize="xx-large")
        # plt.clf()
        # # plt.colorbar()

        plt.subplot(122)
        plt.cla()
        ax = plt.gca()
        ax.imshow((mimg>0).astype(int)+(mimg2>0).astype(int), extent=[-np.pi/2,np.pi/2,-np.pi/2,np.pi/2],origin='lower')
        # ax.contourf(X1r,X2r,mimg,[-100,0,100])
        ax.set_yticks([])
        ax.set_aspect('equal','box')
        ax.set(xlim=(-np.pi/2,np.pi/2),ylim=(-np.pi/2,np.pi/2))

        # plt.show()
        plt.savefig('./demo/hval_demo_a{:02d}.png'.format(i))

    # Sample against model 2 (linear)
    for i in tqdm(range(10)):

        # print(hv.X)
        # print(hv.y)

        xstar = hv2.get_max_perr()
        # print(hv.X)
        # print(xstar)
        ystar2 = compare(xstar,t,nl_compound,lin_simple,sperf)
        # print(ystar)
        hv2.add_data(xstar.reshape(-1,2), ystar2)
    
        plt.subplot(121)
        plt.cla()
        mimg = hv.plot3D(Xr)
        mimg2 = hv2.plot3D(Xr)
        plt.text(-1.5,-1.5,"mdl2",fontsize="xx-large")

        plt.subplot(122)
        plt.cla()
        ax = plt.gca()
        ax.imshow((mimg>0).astype(int)+(mimg2>0).astype(int), extent=[-np.pi/2,np.pi/2,-np.pi/2,np.pi/2],origin='lower')
        # ax.contourf(X1r,X2r,mimg,[-100,0,100])
        ax.set_yticks([])
        ax.set_aspect('equal','box')
        ax.set(xlim=(-np.pi/2,np.pi/2),ylim=(-np.pi/2,np.pi/2))

        # plt.show()
        plt.savefig('./demo/hval_demo_b{:02d}.png'.format(i))