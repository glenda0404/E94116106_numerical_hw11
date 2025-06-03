import math
import matplotlib.pyplot as plt

#==============================#
#      (a) Shooting Method     #
#==============================#
def f_y1(x, z1, z2):
    return [z2, 2*z1 - (x+1)*z2 + (1-x*x)*math.exp(-x)]

def f_y2(x, z1, z2):
    return [z2, 2*z1 - (x+1)*z2]

def rk4_step(x, z, h, f):
    k1 = f(x, z[0], z[1])
    k2 = f(x + h/2, z[0] + h/2*k1[0], z[1] + h/2*k1[1])
    k3 = f(x + h/2, z[0] + h/2*k2[0], z[1] + h/2*k2[1])
    k4 = f(x + h, z[0] + h*k3[0], z[1] + h*k3[1])
    return [
        z[0] + (h/6)*(k1[0]+2*k2[0]+2*k3[0]+k4[0]),
        z[1] + (h/6)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    ]

def shooting_method(h):
    a, b = 0.0, 1.0
    n = int((b - a) / h)
    x_vals = [a + i*h for i in range(n + 1)]
    z1 = [1.0, 0.0]
    z2 = [0.0, 1.0]
    y1_vals, y2_vals = [], []
    for x in x_vals:
        y1_vals.append(z1[0])
        y2_vals.append(z2[0])
        z1 = rk4_step(x, z1, h, f_y1)
        z2 = rk4_step(x, z2, h, f_y2)
    c = (2.0 - y1_vals[-1]) / y2_vals[-1]
    y = [y1 + c*y2 for y1, y2 in zip(y1_vals, y2_vals)]
    return x_vals, y

#==============================#
# (b) Finite Difference Method #
#==============================#
def p(x): return -(x + 1)
def q(x): return 2
def r(x): return (1 - x*x)*math.exp(-x)

def tridiagonal_solver(a, b, c, d):
    n = len(d)
    cp, dp = [0.0]*n, [0.0]*n
    x = [0.0]*n
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / denom if i < n-1 else 0
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
    x[n-1] = dp[n-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

def finite_difference_method(h):
    a, b = 0.0, 1.0
    n = int((b - a) / h) - 1
    x_vals = [i*h for i in range(n+2)]
    y0, yn1 = 1.0, 2.0
    lower, diag, upper, F = [0.0]*n, [0.0]*n, [0.0]*n, [0.0]*n
    for i in range(n):
        xi = x_vals[i+1]
        pi, qi, ri = p(xi), q(xi), r(xi)
        diag[i] = 2 + h**2 * qi
        if i > 0: lower[i] = -(1 + h/2 * pi)
        if i < n-1: upper[i] = -(1 - h/2 * pi)
        F[i] = -h**2 * ri
        if i == 0: F[i] += (1 + h/2 * pi) * y0
        if i == n-1: F[i] += (1 - h/2 * pi) * yn1
    y_inner = tridiagonal_solver(lower, diag, upper, F)
    return x_vals, [y0] + y_inner + [yn1]

#==============================#
#     (c) Variation Method     #
#==============================#
def y1(x): return 1 + x
def phi(x, i): return math.sin(i * math.pi * x)
def phi_prime(x, i): return i * math.pi * math.cos(i * math.pi * x)

def trapz(f, a, b, n):
    h = (b - a) / n
    return h * (0.5*f(a) + sum(f(a + i*h) for i in range(1, n)) + 0.5*f(b))

def variation_method(h):
    a, b = 0.0, 1.0
    N = 5
    n_pts = int((b - a) / h) + 1
    x_vals = [i*h for i in range(n_pts)]
    A = [[0.0]*N for _ in range(N)]
    B = [0.0]*N
    for i in range(N):
        for j in range(N):
            A[i][j] = trapz(lambda x: p(x)*phi_prime(x,i+1)*phi_prime(x,j+1) + q(x)*phi(x,i+1)*phi(x,j+1), a, b, n_pts)
        B[i] = trapz(lambda x: r(x)*phi(x,i+1), a, b, n_pts)
    c = [0.0]*N
    for i in range(N):
        for j in range(i+1, N):
            factor = A[j][i] / A[i][i]
            for k in range(i, N):
                A[j][k] -= factor * A[i][k]
            B[j] -= factor * B[i]
    for i in range(N-1, -1, -1):
        c[i] = B[i]
        for j in range(i+1, N):
            c[i] -= A[i][j] * c[j]
        c[i] /= A[i][i]
    def y(x): return y1(x) + sum(c[i] * phi(x, i+1) for i in range(N))
    return x_vals, [y(x) for x in x_vals]

#==============================#
#         Main + Plot          #
#==============================#
if __name__ == "__main__":
    h = 0.1
    x1, y_shoot = shooting_method(h)
    x2, y_fd = finite_difference_method(h)
    x3, y_var = variation_method(h)

    print("x\tShooting\tFiniteDiff\tVariation")
    print("-"*50)
    for x, ys, yf, yv in zip(x1, y_shoot, y_fd, y_var):
        print(f"{x:.1f}\t{ys:.6f}\t{yf:.6f}\t{yv:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(x1, y_shoot, 'o-', label='Shooting', color='blue')
    plt.plot(x2, y_fd, 's--', label='Finite Diff', color='purple')
    plt.plot(x3, y_var, '^-', label='Variation', color='red')
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Three Methods for Solving BVP")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
