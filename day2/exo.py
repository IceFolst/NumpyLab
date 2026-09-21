import numpy as np

x = np.array([10, 20, 30, 40])

print(x)

x = [1, 2, 3]

print(x * 2)

x = np.array([1, 2, 3])

print(x * 2)

print(np.zeros(5))
print(np.ones(5))
print(np.zeros((3, 4)))
print(np.arange(10))
print(np.arange(2, 10))
print(np.arange(0, 10, 2))

L = np.arange(0, 10)
print(L)
L = np.arange(5, 31, 5)
print(L)

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(A.shape)
print(A.ndim)
print(A.size)

A = np.array([
    [5, 8, 2, 1],
    [9, 3, 7, 4],
    [6, 0, 2, 8]
])

print(A.shape) #(3, 4)
print(A.ndim) #2
print(A.size) #12

A = np.array([
    [3, 8, 1, 9],
    [4, 7, 2, 6],
    [5, 0, 3, 8]
])

print(A[1, :])
print(A[:, 3])

A = np.arange(20).reshape(4, 5)

print(A[1, :])
print(A[:, 2])
print(A[1:3, 1:3])
print(A[2:, :])

x = np.arange(12)
# shape (12,)

A = x.reshape(3, 4)

x = np.arange(24)
# (6, 4) good
# (3, 8) good
# (2, 3, 4) good
# (4, 4) not good
# (1, 24) good
# (24, 1) good

x = np.array([1, 2, 3, 4])

x + 10

x = np.array([2, 5, 8, 3])
print(x*2)
print(x**2)
print(x+5)

x = np.array([3, 12, 7, 20, 4, 15, 1])
print(x[(x > 5) & (x < 18)])
print((x > 5) & (x < 18))

x = np.array([-3, 5, -1, 8, -7, 2])
x[x < 0] = 0
print(x)

x = np.array([1, 2, 3], dtype=np.float32)


A = np.arange(1, 26).reshape(5, 5)
print(A)

print(A[2, :]) # 3rd row
print(A[:, 1]) # 2nd col
print(A[-1, -1])
print(A[1:3, 1:3])
print(A[3: ,:])
print(A[A > 15])
print(A[(A > 10) & (A < 20)])
B = A.copy()
B[A % 2 == 0] = 0
print(B)
print(A)
# A shape is (5, 5)
# A ndim is 2
# A size is 25

X = np.zeros((32, 100, 256))
# X.shape = (32, 100, 256)
# X[0]
# X[0, 0]
# X[:, 0, :]
# X[:, :, 0]
# X[0:5]
# X[:, 10:20, :]
print(X[0].shape)
print(X[0, 0].shape)
print(X[:, 0, :].shape)
print(X[:, :, 0].shape)
print(X[0:5].shape)
print(X[:, 10:20, :].shape)