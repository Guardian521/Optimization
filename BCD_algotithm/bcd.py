import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)
A = np.random.rand(500, 100)
x_ = np.zeros([100, 1])
x_[:5, 0] += np.array([i + 1 for i in range(5)])
b = np.matmul(A, x_) + np.random.randn(500, 1) * 0.1
lam = 10


def lasso(A, x, b, mu):
    residual_term = 0.5 * np.linalg.norm(A @ x - b, ord=2) ** 2
    regularization_term = mu * np.linalg.norm(x, ord=1)
    return residual_term + regularization_term


def BCD(A, x, b, mu):
    f_list = [lasso(A, x, b, mu)]

    for _ in range(100):
        for i in range(len(x)):
            if x[i][0] > 0:
                x[i][0] = 1 / (A[:, i].T @ A[:, i]) * (A[:, i].T @ b2(A, x, b, i) - mu)
            elif x[i][0] < 0:
                x[i][0] = 1 / (A[:, i].T @ A[:, i]) * (A[:, i].T @ b2(A, x, b, i) + mu)
            elif abs(A[:, i].T @ b2(A, x, b, i)) <= mu:
                x[i][0] = 0

        f_list.append(lasso(A, x, b, mu))

    plt.scatter(list(range(len(f_list))), f_list, s=5)
    plt.show()
    print("最终的迭代结果是:", lasso(A, x, b, mu))


def b2(A, x, b, n):
    total = np.zeros((500, 1))

    for i in range(n):
        total += x[i][0] * A[:, i]

    for i in range(n + 1, len(x)):
        total += x[i][0] * A[:, i]

    return b - total


A = np.matrix(A)
BCD(A, x_, b, 10)
BCD(A, x_, b, 1)
BCD(A, x_, b, 0.1)