from math import sin
import numpy as np
from scipy.interpolate import lagrange


def L2Norm(x,y):
    return np.sqrt(np.sum((x-y)**2))

train_x = [0]*20
train_y = [0]*20

for i in range(20):
    # print(i)
    train_x[i] = np.random.uniform(0,20)
    train_y[i] = sin(train_x[i])

poly = lagrange(train_x,train_y)

test_x = [0]*20
test_y = [0]*20

for i in range(20):
    # print(i)
    test_x[i] = np.random.uniform(0,20)
    test_y[i] = sin(test_x[i])

print()
print("L2 Norm of Training Data: ")
print(L2Norm(poly(train_x),train_y))
print("L2 Norm of Testing Data: ")
print(L2Norm(poly(test_x),test_y), "\n")
# print(poly(train_x))

for sigma in range(1,5):

    mu = 0
    s = np.random.normal(mu, sigma, 20)

    print("L2 Norm of Testing Data with sigma = ", sigma)
    test_x = test_x + s
    print(L2Norm(poly(test_x),test_y), "\n")

