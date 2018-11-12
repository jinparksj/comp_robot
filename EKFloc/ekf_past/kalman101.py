import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.mlab as mlab


#1. Expectation, VARIANCE, Covariance, Correlation

x = [3, 1, 2]
p = [0.1, 0.3, 0.4]
E_x = np.sum(np.multiply(x, p))
print(E_x) #1.4

x = np.random.randn(10)
print('variance: ', np.var(x))

x1 = np.random.random((3,3))
print('covariance: ', np.cov(x1))

x_cor = np.random.randn(1, 10)
y_cor = np.random.randn(1, 10) #n = 10

print('covaraiance: ', np.cov(x_cor, y_cor))
print('covaraiance: ', np.cov(x_cor, y_cor, bias=1))

#2. Gaussians
#central limit theorem

a = np.zeros((100, ))
for i in range(100):
    x = [random.uniform(1, 10) for _ in range(1000)]
    a[i] = np.sum(x, axis=0)/1000
plt.figure(1)
plt.hist(a)
plt.grid()

#Gaussian ditribution

mu = 0
vari = 5
sigma = 0.05
x = np.linspace(mu - 5*sigma, mu + 5 * sigma, 100)
plt.figure(2)
plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma)) #pdf: probability density function, f(x), arguments: x, mean, standard deviation
plt.grid()

# Gaussian probability distribution allows us to drive the robots using only one mode with peak at the mean with some variance.

# help(scipy.stats.norm.pdf)

#Gaussian properties
#multiplication
#new mean is mu_new = var *

mu1 = 0 #mean1
variance1= 2 #var1 = 2
sigma1 = np.sqrt(variance1)
x1 = np.linspace(mu1 - 3 * sigma1, mu1 + 3 * sigma1, 100)
plt.figure(3)
plt.plot(x1, scipy.stats.norm.pdf(x1, mu1, sigma1), label = 'prior')

mu2 = 10
variance2 = 2
sigma2 = np.sqrt(variance2)
x2 = np.linspace(mu2 - 3 * sigma2, mu2 + 3 * sigma2, 100)
plt.plot(x2, scipy.stats.norm.pdf(x2, mu2, sigma2), label = 'measurements')

mu_new = (mu1 * variance2 + mu2 * variance1) / (variance1 + variance2)
print('New mean is at: ', mu_new)

var_new = (variance1 * variance2) / (variance1 + variance2)
print('New variance is: ', var_new)

sigma = np.sqrt(var_new)

x3 = np.linspace(mu_new - 3*sigma, mu_new + 3*sigma, 100)
plt.plot(x3, scipy.stats.norm.pdf(x3, mu_new, var_new), label = 'posterior')
plt.legend(loc = 'upper left')
plt.xlim(-10, 20)
plt.grid()

#addition

mu_new_add = mu1 + mu2
print('New mean is at: ', mu_new_add)

var_new_add = variance1 + variance2
print('New variance is: ', var_new_add)
sigma_add=np.sqrt(var_new_add)

x4 = np.linspace(mu_new_add - 3*sigma_add, mu_new_add + 3*sigma_add, 100)
plt.figure(4)
plt.plot(x1, scipy.stats.norm.pdf(x1, mu1, sigma1), label = 'prior')
plt.plot(x2, scipy.stats.norm.pdf(x2, mu2, sigma2), label = 'measurements')

plt.plot(x4, scipy.stats.norm.pdf(x4, mu_new_add, sigma_add), label = 'posterior')
plt.legend(loc = 'upper left')
plt.xlim(-10, 20)
plt.grid()


# numpy einsum example
a = np.arange(25).reshape(5, 5)
b = np.arange(5)
c = np.arange(6).reshape(2, 3)

print('a is', a)
print('b is', b)
print('c is', c)

#This is diagonal sum.

print(np.einsum('ij', a))


plt.show()







