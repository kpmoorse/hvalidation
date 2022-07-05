import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as skg

class HVal:

    def __init__(self, alpha=1e-10):

        self.gpa = skg.GaussianProcessRegressor(alpha=alpha)
        self.X = np.empty((0,1))
        self.y = np.empty(0)

    def fit(self):

        self.gpa.fit(X,y)

    def add_data(self,X,y,fit=True):

        if self.X.size == 0:
            self.X = X
            self.y = y
        else:
            self.X = np.vstack((self.X,X))
            self.y = np.vstack((self.y,y))

        if fit:
            self.fit()

    def plot2D(self, x):

        X = x.reshape(-1,1)

        mean, stdev = self.gpa.predict(X, return_std=True)
        stdev = stdev.reshape(-1,1)
        
        plt.plot(x, mean, 'b-')
        plt.plot(x, mean-stdev, 'b--', x, mean+stdev, 'b--')
        plt.hlines(0,np.min(x),np.max(x),'k','--')

if __name__ == '__main__':

    # Generate test data
    N = 5
    xrange = (-1,1)
    noise = 0.01

    x_full = np.arange(xrange[0],xrange[1],0.01)

    xx = np.random.uniform(xrange[0],xrange[1],N)
    X = xx.reshape(-1,1)

    f_true = lambda x: -x**2+0.5
    y = f_true(X) + np.random.normal(0,noise,(N,1))

    hv = HVal(alpha=noise)
    hv.add_data(X,y)
    hv.plot(x_full)
    plt.plot2D(x_full, f_true(x_full), 'r-')
    plt.show()