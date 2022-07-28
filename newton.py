import numpy as np
import matplotlib.pyplot as plt

class Newton:

    def __init__(self, f):

        self.f = f
        self.dimX = 1

    def query(self,x,info='mean'):

        return self.f(x)

    def get_hessian(self,x,h=1e-5):

        x = x.reshape(-1,1)
        y0 = self.query(x,info='mean')

        hess = []
        for n in range(self.dimX):

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

    def get_rectified_hessian(self,x,h=1e-5):

        x = x.reshape(-1,1)
        y0 = self.query(x,info='mean')**2

        hess = []
        for n in range(self.dimX):

            f = lambda x: self.query(x,info='mean')**2

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

    def get_gradient(self,x,h=1e-5):

        grad = []
        x0 = x.reshape(-1,1)

        for n in range(self.dimX):
            ind = np.arange(self.dimX)==n
            xp = (x+h*ind).reshape(-1,1)
            xm = (x-h*ind).reshape(-1,1)
            dfdxn = (self.query(xp) - self.query(xm))/(2*h)
            grad.append(dfdxn)

        return np.array(grad).reshape(-1,1)

    def get_rectified_gradient(self,x,h=1e-5):

        grad = []
        x0 = x.reshape(-1,1)

        f = lambda x: self.query(x,info='mean')**2

        for n in range(self.dimX):
            ind = np.arange(self.dimX)==n
            xp = (x+h*ind).reshape(-1,1)
            xm = (x-h*ind).reshape(-1,1)
            dfdxn = (f(xp) - f(xm))/(2*h)
            grad.append(dfdxn)

        return np.array(grad).reshape(-1,1)

    # Multivariate Newton's Method
    def get_nearest_optimum(self, x, alpha=1e-5, max_iter=20):

        #via Newton's method
        x = x.reshape(-1,1)
        iter = 0
        while iter < max_iter:

            # Approximate Hessian and gradient via finite difference
            hess = self.get_hessian(x)
            grad = self.get_gradient(x)
            inv_hess = np.linalg.pinv(hess)
            
            x -= inv_hess.dot(grad)

            y = self.query(x,info='mean')
            if np.abs(y) < alpha:
                return x
            else:
                iter += 1
        
        print("Maximum iterations reached")
        return x

    # Multivariate Newton's Method over f^2(x)
    def get_nearest_zero(self, x, alpha=0.5, term=1e-5, max_iter=20):

        #via Newton's method
        x = x.reshape(-1,1)
        iter = 0
        while iter < max_iter:

            # Approximate Hessian and gradient via finite difference
            hess = self.get_rectified_hessian(x)
            grad = self.get_rectified_gradient(x)
            inv_hess = np.linalg.pinv(hess)
            
            # print(x,alpha*inv_hess.dot(grad))
            x -= alpha*inv_hess.dot(grad)
            

            y = self.query(x,info='mean')
            if np.abs(y) < term:
                return x
            else:
                iter += 1
        
        print("Maximum iterations reached")
        return x

# Newton Conjugate Gradient Method


if __name__ == '__main__':

    f = lambda x: 2 - x**2
    newt = Newton(f)

    x = np.array([1.5])
    grad = newt.get_gradient(x)
    print(grad)
    hess = newt.get_hessian(x)
    print(hess)
    print(np.linalg.pinv(hess))
    zero = newt.get_nearest_zero(x)
    print(zero)