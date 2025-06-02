import numpy as np
from scipy.integrate import quad

# a. Shooting Method
def f(x, y):
    y1, y2 = y
    dy1 = y2
    dy2 = -(x + 1) * y2 + 2 * y1 + (1 - x**2) * np.exp(-x)
    return np.array([dy1, dy2])

def runge_kutta(f, y0, x):
    n = len(x)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    h = x[1] - x[0]
    for i in range(1, n):
        k1 = f(x[i-1], y[i-1])
        k2 = f(x[i-1] + h/2, y[i-1] + h/2 * k1)
        k3 = f(x[i-1] + h/2, y[i-1] + h/2 * k2)
        k4 = f(x[i-1] + h, y[i-1] + h * k3)
        y[i] = y[i-1] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

def shooting_method(s1, s2, y0, x, target):
    y1 = runge_kutta(f, [y0, s1], x)
    y2 = runge_kutta(f, [y0, s2], x)
    for _ in range(10):
        c = s2 + (target - y2[-1, 0]) * (s2 - s1) / (y2[-1, 0] - y1[-1, 0])
        y3 = runge_kutta(f, [y0, c], x)
        if abs(y3[-1, 0] - target) < 1e-6:
            return y3[:, 0]
        s1, s2 = s2, c
        y1, y2 = y2, y3
    return y3[:, 0]

# b. Finite-Difference Method
def finite_difference(n, h):
    x = np.linspace(0, 1, n)
    A = np.zeros((n, n))
    F = np.zeros(n)
    A[0, 0] = 1
    F[0] = 1
    A[-1, -1] = 1
    F[-1] = 2
    for i in range(1, n-1):
        xi = x[i]
        pi = -(xi + 1)
        qi = 2
        ri = (1 - xi**2) * np.exp(-xi)
        A[i, i-1] = 1 - (h/2) * pi
        A[i, i]   = -2 + h**2 * qi
        A[i, i+1] = 1 + (h/2) * pi
        F[i] = -h**2 * ri
    return np.linalg.solve(A, F)

# c. Variation Approach
def variation_approach(N=3):
    def phi(i, x): return np.sin(i * np.pi * x)
    def dphi(i, x): return i * np.pi * np.cos(i * np.pi * x)

    A = np.zeros((N, N))
    b = np.zeros(N)
    for i in range(1, N+1):
        for j in range(1, N+1):
            A[i-1, j-1], _ = quad(lambda x: dphi(i,x)*dphi(j,x) + 2*phi(i,x)*phi(j,x), 0, 1)
        b[i-1], _ = quad(lambda x: ((1 - x**2)*np.exp(-x) + 2*(1 - x))*phi(i,x), 0, 1)

    c = np.linalg.solve(A, b)
    x = np.linspace(0, 1, 11)
    y = 1 + x
    for i in range(1, N+1):
        y += c[i-1] * phi(i, x)
    return y

# 執行所有方法
x = np.linspace(0, 1, 11)
y_shooting = shooting_method(0, 1, 1, x, 2)
y_fd = finite_difference(len(x), 0.1)
y_variation = variation_approach(N=3)

# 輸出 a. b. c. 小題結果
print("a. Shooting method result:")
for i in range(len(x)):
    print(f"x = {x[i]:.1f}, y ≈ {y_shooting[i]:.6f}")

print("\nb. Finite-difference method result:")
for i in range(len(x)):
    print(f"x = {x[i]:.1f}, y ≈ {y_fd[i]:.6f}")

print("\nc. Variation approach result:")
for i in range(len(x)):
    print(f"x = {x[i]:.1f}, y ≈ {y_variation[i]:.6f}")
