import numpy as np
import matplotlib.pyplot as plt

def function(x, y, w):
    n = x.shape[0]
    l = 0.1
    J = l * w.dot(w)
    for x_i, y_i in zip(x, y):
        J += np.log(1 + np.exp(-y_i * w.dot(x_i))) / n
    return J

def getGradient(x, y, w):
    n = x.shape[0]
    l = 0.1
    gradient = 2 * l * w
    for x_i, y_i in zip(x, y):
        gradient += (np.exp(-y_i * w.dot(x_i)) * -y_i * x_i) / ((1 + np.exp(-y_i * w.dot(x_i))) * n)
    return gradient

def SGM(x, y):
    alpha = 0.1
    epoch = 50
    np.random.seed(seed=128)
    w = np.random.rand(x.shape[1])
    loss_hist_batch = np.zeros(epoch+1)
    for i in range(epoch):
        loss_hist_batch[i] = function(x, y, w)
        w = w - alpha * getGradient(x, y, w)
    loss_hist_batch[-1] = function(x, y, w)
    return loss_hist_batch

def getHessian(x, y, w):
    n = x.shape[0]
    l = 0.1
    hessian = 2 * l * np.eye(x.shape[1])
    for x_i, y_i in zip(x, y):
        hessian += (np.exp(-y_i * w.dot(x_i)) * (np.outer(x_i, x_i))) / ((1 + np.exp(-y_i * w.dot(x_i))) ** 2) / n
    return hessian

def Newton(x, y):
    alpha = 0.1
    epoch = 50
    np.random.seed(seed=128)
    w = np.random.rand(x.shape[1])
    loss_hist_newton = np.zeros(epoch+1)
    for i in range(epoch):
        loss_hist_newton[i] = function(x, y, w)
        gradient = getGradient(x, y, w)
        hessian = getHessian(x, y, w)
        d = np.linalg.inv(hessian).dot(gradient)
        w = w - alpha * d
    loss_hist_newton[-1] = function(x, y, w)
    return loss_hist_newton

n = 200

np.random.seed(seed=32)
x_d4 = 3 * (np.random.rand(n, 4) - 0.5)
y_d4 = (2 * x_d4[:, 0] - 1 * x_d4[:, 1] + 0.5 + 0.5 * np.random.randn(n)) > 0
y_d4 = 2 * y_d4 - 1

loss_hist_batch = SGM(x_d4, y_d4)
loss_hist_newton = Newton(x_d4, y_d4)
for i in range(len(loss_hist_newton)):
    loss_hist_batch[i] = abs(loss_hist_batch[i] - loss_hist_batch[-1])
    loss_hist_newton[i] = abs(loss_hist_newton[i] - loss_hist_newton[-1])

fig = plt.figure()
fig.subplots_adjust(wspace=0.4, hspace=0.6)
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(loss_hist_batch, label='Steepest Gradient')
ax1.set_yscale('log')
ax1.plot(loss_hist_newton, label='Newton')
ax1.legend()
plt.show()
print('SGM')
for i, loss in enumerate(loss_hist_batch):
    print(f'{i}: {loss - loss_hist_batch[-1]}')
print('Newton')
for i, loss in enumerate(loss_hist_newton):
    print(f'{i}: {loss - loss_hist_newton[-1]}')
