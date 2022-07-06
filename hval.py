import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as skg
import scipy.stats as sps
from tqdm import tqdm

class HVal:

    def __init__(self, alpha=1e-10):

        self.gpa = skg.GaussianProcessRegressor(alpha=alpha)
        self.X = np.empty((0,1))
        self.y = np.empty(0)
        self.dimX = self.X.shape[1]

    def fit(self):

        self.gpa.fit(X,y)
        self.dimX = self.X.shape[1]

    def query(self, x, info="both"):

        if info=="mean":
            return self.gpa.predict(x.reshape(-1,1))
        elif info=="stdev":
            return self.gpa.predict(x.reshape(-1,1), return_std=True)[0]
        elif info=="both":
            return self.gpa.predict(x.reshape(-1,1), return_std=True)

    # Initialize data set or add new data points
    def add_data(self,X,y,fit=True):

        if self.X.size == 0:
            self.X = X
            self.y = y
        else:
            self.X = np.vstack((self.X,X))
            self.y = np.vstack((self.y,y))

        if fit:
            self.fit()

    # Plot mean and stdev of current GPA
    def plot2D(self, x):

        X = x.reshape(-1,1)

        mean, stdev = self.gpa.predict(X, return_std=True)
        stdev = stdev.reshape(-1,1)
        
        plt.plot(x, mean, 'b-')
        plt.plot(x, mean-stdev, 'b--', x, mean+stdev, 'b--')
        plt.hlines(0,np.min(x),np.max(x),'k','--')

    def get_gradient(self,x,dx=1e-5):

        grad = []
        x0 = x.reshape(-1,1)
        y0 = self.gpa.predict(x0)[0][0]
        for n in range(self.dimX):
            ind = np.arange(self.dimX)==n
            xstep = (x+dx*ind).reshape(-1,1)
            dfdxn = (self.gpa.predict(xstep)[0][0] - y0)/dx
            grad.append(dfdxn)

        return np.array(grad)

    def get_hessian(self,x,dx=1e-5):

        hess = []
        for n in range(self.dimX):
            hess.append([])
            ind_n = np.arange(self.dimX)==n
            for m in range(self.dimX):
                ind_m = np.arange(self.dimX)==m
                


    # Very approximate, needs work
    def get_nearest_zero(self, x, alpha=0.001):

        #via Newton's method
        x0 = x
        while True:

        # while True:
        #     grad = self.get_gradient(x)

        #     y = self.gpa.predict(x.reshape(-1,1))

        #     xstep = x - np.sign(y)*grad*alpha

        #     ystep, std_y = self.gpa.predict(xstep.reshape(-1,1), return_std=True)
        #     y, ystep = y.flatten(), ystep.flatten()

        #     std_x = std_y/np.linalg.norm(grad)
        #     net_dist = np.linalg.norm(x0-xstep)
        #     # print(net_dist, std_x*3)
        #     if ystep*y < 0:
        #         return (xstep+x)/2
        #     elif net_dist > std_x*3:
        #         return None
        #     else:
        #         x = xstep
        # pass

    def get_perr(self, x):

        nz = self.get_nearest_zero(x)
        mean, stdev = self.query(x)


        perr = 1-sps.norm.cdf(np.abs(mean/stdev))
        if nz:
            # print("yes")
            zero_std_y = self.query(nz, info="stdev")
            zero_grad = self.get_gradient(nz)
            zero_std_x = zero_std_y/np.linalg.norm(zero_grad)

            nz_dist = np.linalg.norm(nz-x)

            perr *= 1-np.exp(-nz_dist**2/(2*zero_std_x**2))
        else:
            print("no")
        return perr[0]

if __name__ == '__main__':

    # Generate test data
    N = 15
    xrange = (-1,1)
    noise = 0.01

    x_full = np.arange(xrange[0],xrange[1],0.01)

    xx = np.random.uniform(xrange[0],xrange[1],N)
    X = xx.reshape(-1,1)

    f_true = lambda x: -x**2+0.5
    y = f_true(X) + np.random.normal(0,noise,(N,1))

    hv = HVal(alpha=noise)
    hv.add_data(X,y)
    
    # print(hv.get_nearest_zero(np.array([0.5])))

    perr = []
    for i in tqdm(x_full):
        perr.append(hv.get_perr(i))
    perr = np.array(perr)
    # print(perr)
    
    plt.subplot(211)
    hv.plot2D(x_full)
    plt.plot(x_full, f_true(x_full), 'r-')

    plt.subplot(212)
    plt.plot(x_full, perr)

    plt.show()