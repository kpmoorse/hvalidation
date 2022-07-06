import numpy as np
import matplotlib.pyplot as plt
import hval

hv = hval.HVal()

def sim_pendulum(initial_state, t, diffeq, g=9.8, l=1, b=0.1):

    dt = np.diff(t)[0]

    state = [initial_state]

    for k in range(len(t)-1):
        th, thd = state[-1]
        thdd = diffeq(th, thd, (g,l,b))

        state.append([
            th+thd*dt,
            thd+thdd*dt
        ])

    return np.array(state)

def nonlinear(th, thd, params):

    g,l,b = params
    return -g*l*np.sin(th) - b*thd

def linear(th, thd, params):

    g,l,b = params
    return -g*l*th - b*thd

def pperf(state1, state2, thresh):

    # Calculate "periodic distance" between position vectors
    F1 = np.abs(np.fft.fft(state1[:,0]))
    F2 = np.abs(np.fft.fft(state2[:,0]))
    pnorm = np.linalg.norm(F1-F2)
    
    return thresh-pnorm

def compare(initial_state, t, model1, model2, metric, thresh=0):

    state1 = sim_pendulum(initial_state, t, model1)
    state2 = sim_pendulum(initial_state, t, model2)

    return metric(state1, state2, thresh)

if __name__ == '__main__':

    t = np.arange(0,10,0.001)
    th0_list = np.arange(-np.pi/4, np.pi/4, 0.1).reshape(-1,1)
    init_list = np.hstack((th0_list, np.zeros_like(th0_list)))

    g = []
    for i in init_list:
        g.append(compare(i, t, nonlinear, linear, pperf, thresh=50))
    g = np.array(g)

    plt.plot(init_list[:,0], g)
    plt.show()

