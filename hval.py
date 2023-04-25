from sysconfig import get_preferred_scheme
import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as skg
import scipy.stats as sps
import scipy.optimize as spo
from tqdm import tqdm

class HVal(object):

    def __init__(self, alpha=1e-10):

        self.gpa = skg.GaussianProcessRegressor(alpha=alpha)
        self.X = np.empty((0,1))
        self.y = np.empty(0)
        self.dimX = self.X.shape[1]

    def fit(self):

        self.gpa.fit(self.X,self.y)
        self.dimX = self.X.shape[1]

    def query(self, x, info="both"):

        if info=="mean":
            return self.gpa.predict(x.reshape(-1,self.dimX))
        elif info=="stdev":
            return self.gpa.predict(x.reshape(-1,self.dimX), return_std=True)[1]
        elif info=="both":
            return self.gpa.predict(x.reshape(-1,self.dimX), return_std=True)

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


    def get_gradient(self,x,h=1e-5,rect=False):

        grad = []
        x0 = x.reshape(-1,1)

        if rect:
            f = lambda x: self.query(x,info='mean')**2
        else:
            f = lambda x: self.query(x,info='mean')


        for n in range(self.dimX):
            ind = np.arange(self.dimX)==n
            xp = (x+h*ind).reshape(-1,1)
            xm = (x-h*ind).reshape(-1,1)
            dfdxn = (f(xp) - f(xm))/(2*h)
            grad.append(dfdxn)

        return np.array(grad).reshape(-1,1)


    def get_hessian(self,x,h=1e-5,rect=False):

        x = x.reshape(-1,1)
        y0 = self.query(x,info='mean')

        hess = []
        for n in range(self.dimX):

            if rect:
                f = lambda x: self.query(x,info='mean')**2
            else:
                f = lambda x: self.query(x,info='mean')

            hess.append([])
            ind_n = np.arange(self.dimX)==n

            for m in range(self.dimX):

                ind_m = np.arange(self.dimX)==m

                # Calculate finite difference
                if m==n:
                    ind = ind_n # = ind_m
                    fdiff = f(x+ind*h) - 2*f(x) + f(x-ind*h)
                    fdiff /= h**2
                else:
                    fdiff = f(x+ind_n*h+ind_m*h) - f(x+ind_n*h-ind_m*h) - f(x-ind_n*h+ind_m*h) + f(x-ind_n*h-ind_m*h)
                    fdiff /= 4*h**2

                hess[-1].append(fdiff.flatten()[0])

        return np.array(hess)
            

    # Hybrid Powell Method
    def get_nearest_zero(self, x, alpha=0.5, term=1e-5, max_iter=20):

        def f(x):
            val = self.query(x,info='mean')
            return np.repeat(val,self.dimX)
        optim = spo.root(f,x,method='hybr')
        if optim.success:
            root = optim.x
        else:
            root = None
        return root


    def get_perr(self, x, max_exc=0.5, exc=True):

        nz = self.get_nearest_zero(x)
        # nz = self.get_nearest_zero(x)
        # nz = None
        # print(nz)
        mean, stdev = self.query(x,info='both')

        perr = 1-sps.norm.cdf(np.abs(mean/stdev))
        # nz = None
        if (not nz is None) and exc:
            # print("yes")
            zero_std_y = self.query(nz, info="stdev")
            # print(zero_std_y)
            zero_grad = self.get_gradient(nz)
            if np.linalg.norm(zero_grad) == 0:
                # print('zero grad:', nz)
                return 0
            else:
                zero_std_x = zero_std_y/np.linalg.norm(zero_grad)
                zero_std_x = min(zero_std_x,max_exc)
                nz_dist = np.linalg.norm(nz-x)
                perr *= 1-np.exp(-nz_dist**2/(2*zero_std_x**2))

        return perr[0]

    def get_max_perr(self):

        def obj(x):
            return -1*self.get_perr(x)
        b = np.array([[0,10]])
        bounds = np.tile(b,(self.dimX,1))
        optim = spo.differential_evolution(obj, bounds,tol=0.1)

        return optim.x

    # Plot mean and stdev of current GPA
    def plot2D(self, x):

        X = x.reshape(-1,1)
        mean, stdev = self.query(X, info='both')
        
        plt.plot(x, mean, 'b-')
        plt.plot(x, mean-stdev, 'b--', x, mean+stdev, 'b--')
        plt.hlines(0,np.min(x),np.max(x),'k','--')

    def plot3D(self, x):

        # X = x.reshape(-1,self.dimX)
        # print(x)
        mean, stdev = self.query(x, info='both')
        n = int(np.sqrt(x.shape[0]))
        mimg = mean.reshape(n,n)

        plt.imshow(mimg,extent=[0,100,0,100],origin='lower')
        plt.scatter(self.X[:-1,0],self.X[:-1,1],c='k')
        plt.scatter(self.X[-1:,0],self.X[-1:,1],c='w') 

        return mimg  
