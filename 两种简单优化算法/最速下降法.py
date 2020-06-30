import random
import numpy as np
import matplotlib.pyplot as plt
def goldsteinsearch(f, df, d, x, alpham, rho, t):
    flag = 0
    a = 0
    b = alpham
    fk = f(x)
    gk = df(x)
    phi0 = fk
    dphi0 = np.dot(gk, d)
    alpha = b * random.uniform(0, 1)
    while (flag == 0):
        newfk = f(x + alpha * d)
        phi = newfk
        # print(phi,phi0,rho,alpha ,dphi0)
        if (phi - phi0) <= (rho * alpha * dphi0):
            if (phi - phi0) >= ((1 - rho) * alpha * dphi0):
                flag = 1
            else:
                a = alpha
                b = b
                if (b < alpham):
                    alpha = (a + b) / 2
                else:
                    alpha = t * alpha
        else:
            a = a
            b = alpha
            alpha = (a + b) / 2
    return alpha
def rosenbrock(x):
    # 函数:f(x) = 100 * (x(2) - x(1). ^ 2). ^ 2 + (1 - x(1)). ^ 2
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
def jacobian(x):
    # 梯度g(x) = (-400 * (x(2) - x(1) ^ 2) * x(1) - 2 * (1 - x(1)), 200 * (x(2) - x(1) ^ 2)) ^ (T)
    return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])
error=[]
num=[]
def savedata(a,b):
    error.append(a);
    num.append(b)
def steepest(x0):
    print('初始点为:')
    print(x0, '\n')
    imax = 20000
    W = np.zeros((2, imax))
    epo = np.zeros((2, imax))
    W[:, 0] = x0
    i = 1
    x = x0
    grad = jacobian(x)
    delta = sum(grad ** 2)  # 初始误差
    while i < imax and delta > 10 ** (-5):
        p = -jacobian(x)
        x0 = x
        alpha = goldsteinsearch(rosenbrock, jacobian, p, x, 1, 0.1, 2)
        x = x + alpha * p
        W[:, i] = x
        if i % 5 == 0:
            epo[:, i] = np.array((i, delta))
            print(i, np.array((i, delta)))
        grad = jacobian(x)
        delta = sum(grad ** 2)
        i = i + 1
        savedata(np.linalg.norm(grad),i)
    print("迭代次数为:", i)
    print("近似最优解为:")
    print(x, '\n')
def function(x1,x2):
    fff = (100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2)  # 给定的函数
    return fff
if __name__ == "__main__":
    X1 = np.arange(-1.5, 1.5 + 0.05, 0.05)
    X2 = np.arange(-3.5, 4 + 0.05, 0.05)
    [x1, x2] = np.meshgrid(X1, X2)
    f = function(x1,x2)  # 给定的函数
    x0 = np.array([0, 0])
    steepest(x0)
    plt.plot(num, error, 'g*-')  # 画出迭代点收敛的轨迹
    plt.savefig('11111.png')
    plt.show()