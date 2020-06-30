import random
import numpy as np
import matplotlib.pyplot as plt
def goldsteinsearch(f,df,d,x,alpham,rho,t):
  flag = 0
  a = 0
  b = alpham
  fk = f(x)
  gk = df(x)
  phi0 = fk
  dphi0 = np.dot(gk, d)
  alpha=b*random.uniform(0,1)

  while(flag==0):
    newfk = f(x + alpha * d)
    phi = newfk
    # print(phi,phi0,rho,alpha ,dphi0)
    if (phi - phi0 )<= (rho * alpha * dphi0):
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
error=[]
num=[]
def savedata(a,b):
    error.append(a);
    num.append(b)
def frcg(fun,gfun,x0):
  maxk = 5000
  k = 0
  epsilon = 1e-5
  n = np.shape(x0)[0]
  itern = 0
  W = np.zeros((2, 20000))
  while k < maxk:
      W[:, k] = x0
      gk = gfun(x0)
      itern += 1
      itern %= n
      if itern == 1:
        dk = -gk
      else:
        beta = 1.0 * np.dot(gk, gk) / np.dot(g0, g0)
        dk = -gk + beta * d0
        gd = np.dot(gk, dk)
        if gd >= 0.0:
          dk = -gk
      if np.linalg.norm(gk) < epsilon:
        break
      alpha=goldsteinsearch(fun,gfun,dk,x0,1,0.1,2)
      x0+=alpha*dk
      print(k,alpha)
      g0 = gk
      d0 = dk
      k += 1
      savedata(np.linalg.norm(gk), k)
  print("迭代次数为:", k)
  print("近似最优解为:")
  print(x0, '\n')
def fun(x):
  return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
def gfun(x):
  return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])
if __name__=="__main__":
  X1 = np.arange(-1.5, 1.5 + 0.05, 0.05)
  X2 = np.arange(-3.5, 4 + 0.05, 0.05)
  [x1, x2] = np.meshgrid(X1, X2)
  f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2 # 给定的函数
  x0 = np.array([0.0, 0.0])
  frcg(fun,gfun,x0)
  plt.plot(num, error, 'g*-') # 画出迭代点收敛的轨迹
  plt.savefig('22222.png')
  plt.show()