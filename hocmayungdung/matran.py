import numpy as np
np.random.seed(123)

np.random.seed(123)
A = np.random.randint(low=-5, high=5,
size=15).reshape((3,5))
print(A)

A = np.random.uniform(low=-3.5, high=5.5,
size=15).reshape((3,5))
print(A)

A = np.arange(start=-3, stop = 12, dtype =
int).reshape((3,5))
print(A)

A = np.linspace(start=-3, stop=12, num=15,
dtype=float).reshape((3,5))
print(A)

A = np.zeros((3,5))
print(A)

A = np.ones((3,5))
print(A)

A = np.identity(3)
print(A)


