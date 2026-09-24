import numpy as np

'''
Suppose we have:

X.shape = (N, input_dim)

A two-layer neural network can be written as:

Z1 = X @ W1 + b1
A1 = relu(Z1)

Y_pred = A1 @ W2 + b2

Then:
loss = np.mean((Y_pred - Y_true) ** 2)
'''

X = np.array([
    [1., 2.],
    [2., 1.],
    [3., 3.]
]) # shape = (3, 2)

# 3 samples,  2 input features

# Let's use 4 hidden neurons:

input_dim  = 2
hidden_dim = 4
output_dim = 1

np.random.seed(0)

W1 = np.random.randn(2, 4) * 0.1
b1 = np.zeros(4)

W2 = np.random.randn(4, 1) * 0.1
b2 = np.zeros(1)

Y_true = np.array([
    [5.],
    [4.],
    [9.]
]) # shape = (3, 1)

def relu(x):
    return np.maximum(0, x)

Z1 = X @ W1 + b1
A1 = relu(Z1)

Y_pred = A1 @ W2 + b2

loss = np.mean((Y_pred - Y_true) ** 2)

'''
X.shape = (32, 10)

W1.shape = (10, 64)
b1.shape = (64,)

W2.shape = (64, 5)
b2.shape = (5,)

Z1.shape = (32, 64)
A1.shape = (32, 64)
Y_pred.shape = (32, 5)
'''


'''
What backpropagation actually means

Suppose:
Loss = f(Y_pred)
Y_pred = g(A1)
A1 = h(Z1)
Z1 = j(W1)

We want:
dLoss / dW1

But W1 affects the loss through several intermediate operations.

The chain rule says:

dLoss     dLoss   dY_pred   dA1    dZ1
------ = ------- × ------- × ---- × ---
 dW1      dY_pred    dA1     dZ1    dW1

You don't literally multiply those as ordinary scalar numbers in a neural network, because many are tensors/matrices, but conceptually that's what's happening.

Each operation receives a gradient from the operation after it and transforms that gradient backward.
'''



'''
Start from MSE

Our loss is:

L = mean((Y_pred - Y_true)²)

Let:

error = Y_pred - Y_true

Suppose there are N total prediction values.

Then:

dL / dY_pred = 2/N × (Y_pred - Y_true)

In NumPy:

dY_pred = 2 * (Y_pred - Y_true) / Y_pred.size

Shape:

same as Y_pred

This rule is important:
The gradient of an intermediate tensor usually has the same shape as that tensor.
'''


np.random.seed(0)

X = np.array([
    [1., 2.],
    [2., 1.],
    [3., 3.],
    [4., 1.],
    [1., 4.]
])

Y_true = np.array([
    [5.],
    [4.],
    [9.],
    [6.],
    [9.]
])

W1 = np.random.randn(2, 4) * 0.1
b1 = np.zeros(4)

W2 = np.random.randn(4, 1) * 0.1
b2 = np.zeros(1)

learning_rate = 0.01

for step in range(2000):

    # FORWARD
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Y_pred = A1 @ W2 + b2

    error = Y_pred - Y_true
    loss = np.mean(error ** 2)


    # BACKWARD
    dY_pred = 2 * error / Y_pred.size

    dW2 = A1.T @ dY_pred
    db2 = dY_pred.sum(axis=0)

    dA1 = dY_pred @ W2.T

    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    # UPDATE
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if step % 200 == 0:
        print(step, loss)



# MAIN CHALLENGE

np.random.seed(0)

X = np.random.uniform(-2, 2, size=(200, 2))

Y_true = (
    X[:, 0] ** 2
    + 2 * X[:, 1]
).reshape(-1, 1)

input_dim = 2
hidden_dim = 64
output_dim = 1

learning_rate = 0.01

W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros(hidden_dim)

W2 = np.random.randn(hidden_dim, output_dim) * 0.1
b2 = np.zeros(1)

for step in range(5000):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Y_pred = A1 @ W2 + b2

    error = Y_pred - Y_true
    loss = np.mean(error ** 2)


    dY_pred = 2 * error / Y_pred.size

    dW2 = A1.T @ dY_pred
    db2 = dY_pred.sum(axis=0)

    dA1 = dY_pred @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)


    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if step % 500 == 0:
        print(step, loss)


test = np.array([
    [1.5, 1.0],
    [1.0, 1.5],
    [-1.5, 0.5],
    [0.0, -1.0]
])

Z1 = test @ W1 + b1
A1 = relu(Z1)
pred = A1 @ W2 + b2

print(pred)