import numpy as np

# ReLU(x) = max(0, x)

x = np.array([-3., -1., 0., 2., 5.])

def relu(x):
    return np.maximum(0, x)

print(relu(x))

x = np.array([-5., 2., -1., 7., 0., 3.])
print(relu(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-2., 0., 2.])
print(sigmoid(x))

#Important special case:
#  sigmoid(0)   is:   0.5
#  Large positive values approach 1.
#  Large negative values approach 0.

x = np.array([
    -100.,  # near 0
    -5.,    # near 0.5
    0.,     # 0.5
    5.,     # near 0.5
    100.    # near 1
])
#print(sigmoid(x))

#                   eˣᵢ
# softmax(xᵢ) = ----------
#               Σⱼ eˣⱼ

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

scores = np.array([2., 1., 0.])
print(softmax(scores))
print(softmax(scores).sum())

x = np.array([1000., 1001., 1002.])
# print(np.exp(x))  it's overflow

def softmax(x):
    shift = x - np.max(x)
    exp_x = np.exp(shift)
    return exp_x / exp_x.sum()
print(softmax(x))
print(softmax(x).sum())

scores = np.array([
    [2., 1., 0.],
    [1., 3., 2.]
])

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

X = np.array([
    [1., 2., 3.],
    [3., 2., 1.],
    [4., 4., 4.]
])
P = softmax(X)
print(P.sum(axis=1))


Q = np.random.randn(32, 100, 64)
K = np.random.randn(32, 100, 64)
scores = Q @ K.swapaxes(-1, -2)

d = Q.shape[-1]

scores = Q @ K.swapaxes(-1, -2)
scores = scores / np.sqrt(d)

# Q.shape == (16, 128, 32)
# K.shape == (16, 128, 32)
# K.swapaxes(-1, -2).shape is (16, 32, 128)
# (Q @ K.swapaxes(-1, -2)).shape is (16, 128, 128)
# d = Q.shape[-1] => d is 128


Q = np.random.randn(2, 4, 3)
K = np.random.randn(2, 4, 3)

scores = Q @ K.swapaxes(-1, -2)
d = Q.shape[-1]
scaled_scores = scores / np.sqrt(d)
weights = softmax(scaled_scores)
print(weights.shape)
print(weights.sum(axis=-1))


# Q = queries
# K = keys
# V = values

# Q.shape == (32, 100, 64)
# K.shape == (32, 100, 64)
# V.shape == (32, 100, 64)

# weights = softmax(
#     Q @ K.swapaxes(-1, -2) / np.sqrt(64)
# )

# output = weights @ V
# Shape reasoning:
# (32, 100, 100) @ (32, 100, 64)
#
# For each batch:
# (100, 100) @ (100, 64)    produces:    (100, 64)
#
# So overall:
# output.shape = (32, 100, 64)

Q = np.array([
    [1., 0.],
    [0., 1.]
])

K = np.array([
    [1., 0.],
    [0., 1.]
])

V = np.array([
    [10., 0.],
    [0., 20.]
])

scores = Q @ K.swapaxes(-1, -2)
d = Q.shape[-1]
scaled_scores = scores / np.sqrt(d)
weights = softmax(scaled_scores)
output = weights @ V
print(weights[0, 0])
print(weights[0, 1])


Q = np.random.randn(32, 8, 100, 64)
K = np.random.randn(32, 8, 100, 64)
V = np.random.randn(32, 8, 100, 64)
# Shape:
# (batch, heads, tokens, head_dim)

scores = Q @ K.swapaxes(-1, -2)
scores = scores / np.sqrt(Q.shape[-1])
weights = softmax(scores)
output = weights @ V


# Suppose a model predicts:
probs = np.array([0.1, 0.7, 0.2])

# and the correct class is: class 1
#
# The model assigned the correct class probability:  0.7
#
# Cross-entropy loss for this example is roughly:  -log(0.7)
loss = -np.log(0.7)





Q = np.array([
    [1., 0., 1.],
    [0., 1., 1.],
    [1., 1., 0.]
])

K = np.array([
    [1., 1., 0.],
    [1., 0., 1.],
    [0., 1., 1.]
])

V = np.array([
    [10., 0.],
    [0., 20.],
    [5., 5.]
])

def softmax(x):
    shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shift)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
d = Q.shape[-1]

scores = Q @ K.swapaxes(-1, -2)
scaled_scores = scores / np.sqrt(d)
weights = softmax(scaled_scores)
output = weights @ V
print(weights.sum(axis=-1))

def attention(Q, K, V):
    s = Q @ K.swapaxes(-1, -2)
    s = s / np.sqrt(Q.shape[-1])
    w = softmax(s)
    o = w @ V
    return o