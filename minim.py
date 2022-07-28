import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

class ABC(object):

    def __init__(self):

        pass

    def obj(self, x):
        return (x-0.5)**2-2

if __name__ == '__main__':

    abc = ABC()
    optim = spo.brute(abc.obj, [[-1,1]])
    min = optim[0]

    print(min)