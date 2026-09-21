import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(A.sum())
print(A.sum(axis=0))  # squished into one row   []
print(A.sum(axis=1))  # squished into one col  > () <

A = np.array([
    [2, 4, 6, 8],
    [1, 3, 5, 7],
    [10, 20, 30, 40]
])

print(A.sum(axis=0)) # [13, 27, 41, 55]
print(A.sum(axis=1)) # [20, 16, 100]

print(A.mean(axis=0)) # [13/3, 9, 41/3, 55/3]
print(A.mean(axis=1)) # [5, 4, 25]

# np.sum()
# np.mean()  => arithmetic mean on given axe
# np.min()  => return min val at given axe
# np.max()  => return max val at given axe
# np.std()  => compute standard deviation on given axe
#  np.var()  => compute variance on specified axe
#
# np.argmin()  => return min indice at given axe
# np.argmax()  => return max indice at given axe

A = np.array([
    [4, 9, 1],
    [7, 2, 8]
])
print(A.argmax(axis=1)) # give list of max pos of each row, here [1, 2], max at each y
print(A.argmax(axis=0))  # give list of max pos of each col, here [1, 0, 1], max at each x

A = np.array([
    [3, 8, 2, 5],
    [9, 1, 4, 7],
    [2, 6, 10, 3]
])

print(A.max(axis=1)) # [8, 9, 10]
print(A.argmax(axis=1))  # [1, 0, 2]

print(A.max(axis=0))  # [9, 8, 10, 7]
print(A.argmax(axis=0)) # [1, 0, 2, 1]

X = np.array([
    [170, 65, 25],
    [180, 80, 40],
    [160, 55, 30],
    [175, 72, 35]
]) # each row is one person, and the columns are the feature, (height, weight, age)
# X.shape = (4, 3)

print(X.mean(axis=0)) # one mean per feature

X = np.array([
    [10., 100., 5.],
    [20., 200., 7.],
    [30., 300., 9.]
])

print(X.mean(axis=0)) # [20., 200., 7.]
X_centered = X - X.mean(axis=0)
#print(X_centered)

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
]) # shape (2, 3)

b = np.array([10, 20, 30]) # shape (3,)

print(A + b)

# (3, 4) + (4,)  => good, shape = (3, 4)
#
# (3, 4) + (3,) => not good
#
# (3, 4) + (1, 4) => good, shape = (3, 4)
#
# (3, 4) + (3, 1) => good, shape = (3, 4)
#
# (5, 2, 8) + (8,) => good, shape = (5, 2, 8)
#
# (5, 2, 8) + (2, 8) => good, shape = (5, 2, 8)
#
# (5, 2, 8) + (5, 1, 1) => good, shape =  (5, 2, 8)
#
# (5, 2, 8) + (5, 2) => not good

# (3,)    horizontal-like vector
# (3, 1)  column vector



A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

b = np.array([
    [100],
    [200],
    [300]
])
print(A + b)

x = np.arange(12) # has shape = (12,)
x.reshape(3,4)
x.reshape(3,-1) # -1 means NumPy, infer this dim for me
# 3 * ? = 12 => ? = 4

x = np.arange(60)
x.reshape(5, -1) # (5, 12)
x.reshape(-1, 10) # (6, 10)
x.reshape(3, 4, -1) # (3, 4, 5)
x.reshape(2, -1, 5) # (2, 6, 5)
# x.reshape(7, -1) # does not work because 60 / 7 is not a round number

x = np.array([1, 2, 3, 4])
print(x[None, :].shape) # shape = (1, 4)
print(x[:, None].shape) # shape = (4, 1)
print(x.reshape(2, 2).shape) # shape = (2, 2)

# to add one diff val to each col, you need to use shape = (1, 4)
# to add one diff val to each row, you need to use shape = (4, 1)

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
]) # shape = (2, 3)

print(A.T) # shape (3, 2)

A = np.arange(20).reshape(4, 5)
print(A.shape) # shape = (4, 5)
print(A.T.shape) # shape = (5, 4)
print(A.T[0]) # [0, 5, 10, 15]
print(A.T[:, 0]) # [0, 1, 2, 3, 4]

X = np.zeros((32, 128, 64))
print(X.swapaxes(1, 2).shape) # shape = (32, 64, 128)
print(X.transpose(0, 2, 1).shape) # shape = (32, 64, 128)

x = np.array([1, 2, 3, 4, -1])

result = []
for value in x:
    result.append(value * 3 + 2)
print(x * 3 + 2)

result = []
for value in x:
    if value < 0:
        result.append(0)
    else:
        result.append(value)
x[x < 0] = 0
print(x)

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
result = []
for row in A:
    result.append(sum(row))

print(A.sum(axis=1))

X = np.array([
    [10., 200., 3.],
    [20., 400., 7.],
    [30., 300., 5.],
    [40., 500., 9.]
])

mean = X.mean(axis=0)
std = X.std(axis=0)
X_standardized = (X - mean) / std
print(X_standardized)

X = np.array([
    [10., 100.,  5., 50.],
    [20., 200.,  7., 40.],
    [30., 300.,  6., 60.],
    [40., 400., 10., 80.],
    [50., 500.,  2., 70.]
])

print(X.shape) # (5, 4)
mean = X.mean(axis=0)
std = X.std(axis=0)
center = X - mean
stand = (X - mean) / std
max_vals = np.max(X, axis=1)
max_inds = np.argmax(X, axis=1)
min_val = np.min(X, axis=0)
sums = X.sum(axis=1)
print(X > 100)
B = X.copy()
B[B < 10] = 0
print(X.T.shape)

print(stand.mean(axis=0))
print(stand.std(axis=0))

X = np.zeros((32, 100, 256))

print(X.mean(axis=0).shape) # shape = (100, 256)
print(X.mean(axis=1).shape) # shape = (32, 256)
print(X.mean(axis=2).shape) # shape = (32, 100)

print(X.max(axis=-1).shape) # shape = (32, 100)

print(X[:, 0, :].shape) # shape = (32, 256)
print(X[:, :, 0].shape) # shape = (32, 100)

print(X.swapaxes(1, 2).shape) # shape = (32, 256, 100)

print(X.reshape(32, 100, 8, 32).shape) # shape = (32, 100, 8, 32)
