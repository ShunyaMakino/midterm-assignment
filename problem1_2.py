import numpy as np
import matplotlib.pyplot as plt

def function(x, y, W):
    n = x.shape[0]
    l = 0.1
    J = l * np.linalg.norm(W, ord=2)
    for x_i, y_i in zip(x, y):
        J += (-W.dot(y_i).dot(x_i) + np.log(np.sum(np.exp(W.transpose().dot(x_i))))) / n
    return J

def getGradient(x, y, W, c):
    n = x.shape[0]
    l = 0.1
    w_c = W[:, c]
    gradient = 2 * l * w_c
    for x_i, y_i in zip(x, y):
        p = np.exp(np.dot(w_c, x_i)) / np.sum(np.exp(np.dot(W.transpose(), x_i)))
        gradient += (p - y_i[c]) * x_i / n
    return gradient

def SGM(x, y):
    alpha = 0.1
    epoch = 50
    np.random.seed(seed=128)
    W = np.random.rand(x.shape[1], x.shape[1])
    loss_hist_batch = np.zeros(epoch+1)
    for i in range(epoch):
        loss_hist_batch[i] = function(x, y, W)
        for c in range(W.shape[1]):
            W[:,c] = W[:,c] - alpha * getGradient(x, y, W, c)
    loss_hist_batch[-1] = function(x, y, W)
    return loss_hist_batch

def getHessian(x, y, W, c):
    n = x.shape[0]
    l = 0.1
    w_c = W[:, c]
    hessian = 2 * l * np.eye(x.shape[1])
    for x_i, y_i in zip(x, y):
        p = np.exp(np.dot(w_c, x_i)) / np.sum(np.exp(np.dot(W.transpose(), x_i)))
        hessian += np.outer(x_i, x_i) * p * (1 - p) / n
    return hessian

def Newton(x, y):
    alpha = 0.1
    epoch = 50
    np.random.seed(seed=128)
    W = np.random.rand(x.shape[1], x.shape[1])
    loss_hist_newton = np.zeros(epoch+1)
    for i in range(epoch):
        loss_hist_newton[i] = function(x, y, W)
        for c in range(W.shape[1]):
            gradient = getGradient(x, y, W, c)
            hessian = getHessian(x, y, W, c)
            d = np.linalg.inv(hessian).dot(gradient)
            W[:,c] = W[:,c] - alpha * d
    loss_hist_newton[-1] = function(x, y, W)
    return loss_hist_newton


n = 200
np.random.seed(seed=32)
x_d5 = 3 * (np.random.rand(n, 4) - 0.5)
W = np.array([[ 2,  -1, 0.5,],
              [-3,   2,   1,],
              [ 1,   2,   3]])
y_d5 = np.argmax(np.dot(np.hstack([x_d5[:,:2], np.ones((n, 1))]), W.T)
                        + 0.5 * np.random.randn(n, 3), axis=1)

x = np.hstack([x_d5[:,:2], np.ones((n, 1))])
y = np.identity(3)[y_d5]

loss_hist_batch = SGM(x, y)
loss_hist_newton = Newton(x, y)
for i in range(len(loss_hist_batch)):
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

print('SGM:')
for i, loss in enumerate(loss_hist_batch):
    print(f'{i}: {loss}')

print('Newton:')
for i, loss in enumerate(loss_hist_newton):
    print(f'{i}: {loss}')
