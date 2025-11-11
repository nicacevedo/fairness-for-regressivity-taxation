from scipy.optimize import brentq, fsolve
import numpy as np

A,Q,p = np.random.random(size=3)*2
delta = 1
# a,b = np.random.randint(size=2, low=1, high=20)
f = lambda x: -100*A*x + (Q/p**2)*(np.exp(p*x)-p*x-1) + delta

# print(f(0))

# for x0 in [10**(-(i+1)) for i in range(0,4,0.1)]:

i = 1
x0 = delta/A
x_quad = (A-np.sqrt(A**2 - 2*Q*delta))/Q # safe in [0,x_quad]
print(f(0))
print(f(x_quad))


root = fsolve(f, x_quad)
print(root)
print("f in root: ", f(root))


root = brentq(f, 0, x_quad)
print(root)
print("f in root: ", f(root))


# root = -1
# while root <= 0: 
#     # x0 = 1e-2
#     root = fsolve(f, x0)  # x0 is initial guess
#     # root = brentq(f, a, b)
#     print(root)
#     print("f in root: ", f(root))

#     i+=0.1
#     x0 -= 10**(-i)
# print("Nonegative root: ", root)
# print("f in root: ", f(root))