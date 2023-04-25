import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import sklearn.gaussian_process as skg
import scipy.optimize as spo

def f(x):
    val = 1-(x[0]**2+x[1]**2)
    res = np.array([val,val])
    print(res)
    return res
x_test = np.array([0.7,0.7]).reshape(1,-1)
optim = spo.root(f,x_test,method='lm')
print(optim.x)