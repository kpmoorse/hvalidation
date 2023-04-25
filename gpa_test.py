import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import sklearn.gaussian_process as skg
import scipy.optimize as spo

N = 20
xrange = (-1.5,1.5)
noise = 0.01
gpa = skg.GaussianProcessRegressor(alpha=noise)

x1r = np.arange(xrange[0],xrange[1],0.01)
x2r = x1r.copy()
# print(x1r.shape)
X1r, X2r = np.meshgrid(x1r,x2r)
Xr = np.hstack((X1r.reshape(-1,1),X2r.reshape(-1,1)))
# print(Xr.shape)

x1 = np.random.uniform(xrange[0],xrange[1],N)
x2 = np.random.uniform(xrange[0],xrange[1],N)
X = np.hstack((x1.reshape(-1,1),x2.reshape(-1,1)))
f_true = lambda x1,x2: (1-(x1**2+x2**2)).reshape(-1,1)
y = f_true(x1,x2) + np.random.normal(0,noise,(N,1))

gpa.fit(X,y)
mean, std = gpa.predict(Xr, return_std=True)
# print(Xr.shape)
# print(mean.shape)
# plot(*X.T,'.')

x_test = np.array([0,1])[0]
print(gpa.predict(x_test))

def f(x):
    val =  gpa.predict(x)
    return np.tile(val,(1,2))
spo.root(f,x_test)

plt.imshow(mean.reshape(300,300))
plt.show()