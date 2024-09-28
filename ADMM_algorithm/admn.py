import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2021) # set a constant seed to get same random matrixs
A = np.random.rand(500, 100)
x_ = np.zeros([100, 1])
x_[:5, 0] += np.array([i+1 for i in range(5)]) # x_denotes expected x
b = np.matmul(A, x_) + np.random.randn(500, 1) * 0.1 #add a noise to b
lam = 10 # try some different values in {0.1, 1, 10}

def lasso(A, x, z, b, lam):
    f = 0.5 * np.linalg.norm(A @ x - b, ord=2) ** 2 + lam * np.linalg.norm(z, ord=1)
    return f

def Beta(A):
    return max(np.linalg.eig(A.T @ A)[0])

def xp(z, lam, A):
    temp = np.maximum(0, abs(z) - lam / Beta(A))
    return np.sign(z) * temp

def ADMM(A, x, b, lam):
    mu = np.ones([100, 1])
    rho = Beta(A)
    rho_i = np.identity(A.shape[1]) * rho
    z = x
    F = []
    f = lasso(A, x, z, b, lam)

    for k in range(100):
        x = np.linalg.inv(A.T @ A + rho_i) @ (A.T @ b + rho * (z - mu))
        z = xp(x + mu, lam, A)
        mu = mu + x - z
        deltaf = (f - lasso(A, x, z, b, lam)) / lasso(A, x, z, b, lam)
        f = lasso(A, x, z, b, lam)
        F.append(f)

    plt.scatter(list(range(0, 100)), F, s=5)
    plt.show()
    print("最终的迭代结果是：",lasso(A, x, z, b, lam))

np.random.seed(2021)
A = np.random.rand(500, 100)
ADMM(A, x_, b, 10)
ADMM(A, x_, b, 1)
ADMM(A, x_, b, 0.1)