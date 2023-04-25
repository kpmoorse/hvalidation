import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as skg
from scipy.special import erfc
from copy import deepcopy

f = lambda x: 1-2*x**2

gpa = skg.GaussianProcessRegressor(alpha=1e-2)
X = np.random.uniform(-1,1,3).reshape(-1,1).reshape(-1,1)
y = f(X)
gpa.fit(X,y)

xx = np.arange(-1,1,0.001)
mu, sigma = gpa.predict(xx.reshape(-1,1), return_std=True)
sigma = sigma.reshape(-1,1)

plt.subplot(2,1,1)
plt.plot(xx, mu, 'r')
plt.plot(xx,mu-3*sigma,'r--')
plt.plot(xx,mu+3*sigma,'r--')
plt.scatter(X,y,c='k')
plt.hlines(0,-1,1,'k','--')

plt.gca().set_xticks([], [])
plt.ylabel("$v(\phi)$")
plt.legend(["$\mu(\phi)$", "$\mu(\phi) \pm 3\sigma(\phi)$"])

p_err = 1/2*erfc(np.abs(mu)/sigma)

def gpa_next(gpa,x,y):

    g = skg.GaussianProcessRegressor(alpha=1e-2)
    g.fit(
        np.vstack((gpa.X_train_, np.array(x).reshape(-1,1))),
        np.vstack((gpa.y_train_, np.array(y).reshape(-1,1)))
    )
    # print(g.X_train_)
    return g

x_next_list = np.arange(-1,1,0.01)
f = []

for x_next in x_next_list:

    y_next = gpa.predict(x_next.reshape(-1,1))
    g = gpa_next(
        gpa,
        x_next,
        y_next
        )
    mu, sigma = g.predict(xx.reshape(-1,1), return_std=True)
    sigma = sigma.reshape(-1,1)
    p_err_next = 1/2*erfc(np.abs(mu)/sigma)
    f.append(np.sum(p_err-p_err_next))

f = np.array(f)

plt.subplot(2,1,2)
# ax = plt.gca()
# ax2 = ax.twinx()

# ax.plot(xx, p_err, 'g')
# ax2.plot(x_next_list,f, 'b')

# ax.set_xlabel('$\phi$')
# ax.set_ylabel('$P^{err}(\phi)$')
# ax.tick_params(axis='y', colors='g')
# ax.yaxis.label.set_color('g')

# ax2.set_ylabel('$f(\phi)$')
# ax2.tick_params(axis='y', colors='b')
# ax2.yaxis.label.set_color('b')

plt.plot(x_next_list,f, 'b')
plt.ylabel('$f(\phi)$')

plt.show()