import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.dot(a, b))

# 1*4 + 2*5 + 3*6 = 32

print(a @ b) # For 1D vectors, @ gives the dot product.

a = np.array([2, 4, 6])
b = np.array([1, 3, 5])

print(a @ b) # 44

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
]) # shape = (2, 3)

B = np.array([
    [10, 20],
    [30, 40],
    [50, 60]
]) # shape = (3, 2)

C = A @ B # (2, 3) @ (3, 2) => result shape = (2, 2)
# (m, n) @ (n, p) => (m, p)
print(C)

# (4, 3) @ (3, 5) good => (4, 5)
#
# (4, 3) @ (4, 5) not good
#
# (10, 20) @ (20, 7) good => (10, 7)
#
# (32, 128) @ (128, 64) good => (32, 64)
#
# (32, 128) @ (64, 128) not good


A = np.array([
    [1, 0, 2],
    [3, 1, 1]
])

B = np.array([
    [2, 1],
    [1, 0],
    [4, 3]
])

C = A @ B # [ [10, 7], [11, 6] ]
print(C)

x = np.array([2., 5., 1.])
W = np.array([
    [0.5, 1.0],
    [1.5, 0.2],
    [0.3, 2.0]
])
b = np.array([0.1, 0.2])

y = x @ W + b
print(y)

# Suppose
# x.shape = (128,)
# W.shape = (128, 64)
# b.shape = (64,)

# (x @ W).shape is (64,)
# (x @ W + b).shape is (64,)

# Then suppose instead:
# X.shape = (32, 128)
# X @ W shape is (32, 64)
# X @ W + b shape is (32, 64)

X = np.random.randn(32, 128)
W = np.random.randn(128, 64)
Y = X @ W


X = np.array([
    [1., 2., 3.],
    [4., 5., 6.]
])
W = np.array([
    [0.5, 1.0],
    [1.0, 0.0],
    [0.2, 2.0]
])
b = np.array([0.1, -0.5])
Y = X @ W + b
print(Y.shape) # shape = (2, 2)
print(Y)


x = np.array([3., 4.])
# the L2 norm is:  sqrt(3² + 4²) = 5
# np.linalg.norm(x)  returns:  5.0

x = np.array([6., 8.])
print(np.linalg.norm(x)) # should be sqrt(6² + 8²) = 10
x = np.array([1., 2., 2.])
print(np.linalg.norm(x)) # should be 3

X = np.array([
    [3., 4.],
    [5., 12.]
])
np.linalg.norm(X, axis=1) # result is [5, 13],

X = np.array([
    [3., 4., 0.],
    [1., 2., 2.],
    [6., 0., 8.]
])
print(np.linalg.norm(X, axis=1)) # [5, 3, 10]

x = np.array([3., 4.]) # the norm is 5
x_normalized = x / np.linalg.norm(x)
print(x_normalized)  # it's new norm is: 1.


X = np.random.randn(100, 256)
norms = np.linalg.norm(X, axis=1) # has shape = (100,)
# But X has shape:  (100, 256)
# So:  X / norms  will not broadcast correctly.
norms = np.linalg.norm(X, axis=1, keepdims=True) # has shape(100, 1)
X_normalized = X / norms # this works now
print(X_normalized)



X = np.random.randn(32, 128)
print(X.mean(axis=1).shape) # shape = (32,)
print(X.mean(axis=1, keepdims=True).shape) # shape = (32, 1)

X - X.mean(axis=1, keepdims=True) # is useful, because you sub each mean val for each col


A = np.random.randn(32, 100, 64)
B = np.random.randn(32, 64, 50)
C = A @ B # shape = (32, 100, 50)

# For each batch independently:
#
# (100, 64) @ (64, 50)
#        ↓
#     (100, 50)


Q = np.random.randn(32, 128, 64)
K = np.random.randn(32, 128, 64)
# 32 batches
# 128 tokens
# 64-dimensional vectors

# We want each token to compare with every other token.
# Q shape: (32, 128, 64)
# To matrix multiply, we want K shaped: (32, 64, 128)
K_T = K.swapaxes(-1, -2)
scores = Q @ K_T # shape = (32, 128, 128)


# Q.shape = (16, 100, 32)
# K.shape = (16, 100, 32)
# K.swapaxes(-1, -2).shape is (16, 32, 100)
# (Q @ K.swapaxes(-1, -2)).shape is (16, 100, 100)


# Suppose:
# X = np.random.randn(32, 100, 256)
# Shape:
# (batch, tokens, embedding)

# We want 8 attention heads.
# Since: 256 / 8 = 32
#  X_heads = X.reshape(32, 100, 8, 32)
# Shape:
# (batch, tokens, heads, head_dim)

# Sometimes we want:
# (batch, heads, tokens, head_dim)
#  X_heads = X_heads.transpose(0, 2, 1, 3)
# Now shape:
# (32, 8, 100, 32)


X = np.zeros((64, 128, 512))
head_dim = X.reshape(64, 128, 8, 64)
print(head_dim.transpose(0, 2, 1, 3).shape) # is (64, 8, 128, 64)



X = np.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.],
    [2., 4., 6.]
])
W = np.array([
    [0.5, 1.0],
    [1.5, 0.0],
    [0.2, 2.0]
])
b = np.array([0.1, -0.2])

# 1. What are X.shape, W.shape and b.shape?
# X.shape = (4, 3)      W.shape = (3, 2)        b.shape = (2,)

# 2. Is X @ W valid? Why?
# (4, 3) @ (3, 2) => (4, 2) cause (n, m) @ (m, p) => (n, p)

Y = X @ W + b
row_norm = np.linalg.norm(X, axis=1, keepdims=True)

X_normalized = X / row_norm
print(np.linalg.norm(X_normalized, axis=1))

largest_row_norm = np.argmax(row_norm)
print(largest_row_norm)

print(X.T @ X) # shape is (3, 3)
print(X @ X.T) # shape is (4, 4)


Q = np.random.randn(32, 8, 100, 64)
K = np.random.randn(32, 8, 100, 64)
# Interpret shape as:
# (batch, heads, tokens, head_dim)

K_T = K.swapaxes(-1, -2)
scores = Q @ K_T
print(K_T.shape) # shape = (32, 8, 64, 100)
print(scores.shape) # shape = (32, 8, 100, 100)