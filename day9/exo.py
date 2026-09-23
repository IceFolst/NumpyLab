import numpy as np

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

y_true = np.array([2., 4.])
y_pred = np.array([1., 5.])
print(mse(y_pred, y_true))

y_true = np.array([2., 4., 6.])
y_pred = np.array([1., 5., 4.])
# error = [-1, 1, -2]
# squared = [1, 1, 4]
# mean = 2
print(mse(y_pred, y_true))

'''
Suppose:
loss = f(w)

The derivative tells us:
If I slightly increase w, which way does the loss move?

If:
dLoss/dw > 0
then increasing w increases the loss.

So we should decrease w.

If:
dLoss/dw < 0
then increasing w decreases the loss.

So we should increase w.

This leads to:

w_new = w - learning_rate × gradient

This is gradient descent.
'''



'''
Model:
prediction = wx

Loss:
L = mean((wx - y)²)

The derivative with respect to w is:
dL/dw = mean(2(wx - y)x)

gradient = np.mean(2 * (y_pred - y_true) * x)
'''

x = np.array([1., 2., 3., 4.])
y_true = np.array([2., 4., 6., 8.])

w = 0.5
learning_rate = 0.01

for step in range(100):
    y_pred = w * x

    loss = np.mean((y_pred - y_true) ** 2)

    gradient = np.mean(
        2 * (y_pred - y_true) * x
    )

    w = w - learning_rate * gradient

    if step % 10 == 0:
        print(step, loss, w)


'''
Suppose the true function is:
y = 3x + 2
'''

x = np.array([0., 1., 2., 3., 4.])
y_true = np.array([2., 5., 8., 11., 14.])

w = 0.0
b = 0.

learning_rate = 0.01

'''
Implement the complete training loop for:
y = 3x + 2
'''

for step in range(1000):
    y_pred = w * x + b

    loss = np.mean((y_pred - y_true) ** 2)

    error = (y_pred - y_true)

    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)

    w -= learning_rate * dw
    b -= learning_rate * db

    if step % 100 == 0:
        print(step, loss, w, b)



X = np.array([
    [1., 2.],
    [2., 1.],
    [3., 4.],
    [4., 3.]
]) # shape = (4, 2)
W = np.array([0., 0.]) # shape = (2,)
y_pred = X @ W # shape = (4,)


'''
For:
y_pred = XW

the gradient is:

dW = (2 / N) * X.T @ (y_pred - y_true)


Suppose:
X              (4, 2)
X.T            (2, 4)

error          (4,)

Then:
X.T @ error

(2, 4) @ (4,)
    ↓
   (2,)

Same shape as:
W = (2,)

This is not accidental.

A gradient must have the same shape as the parameter it updates.
That's an important rule.
'''

# Suppose:
#
# X.shape = (100, 20)
# W.shape = (20,)
# y.shape = (100,)

# Predict:
#
# (X @ W).shape = (100,)
#
# error.shape = (100,)
#
# (X.T @ error).shape = (20,), the same shape as the parameter it updates

X = np.array([
    [1., 0.],
    [0., 1.],
    [1., 1.],
    [2., 1.],
    [1., 2.]
])

# True relationship:
# y = 2*x1 + 3*x2

y_true = np.array([2., 3., 5., 7., 8.])

W = np.zeros(2)

learning_rate = 0.01

for step in range(1000):
    y_pred = X @ W

    error = (y_pred - y_true)

    loss = np.mean(error ** 2)

    gradient = (2 / len(X)) * X.T @ error

    W -= learning_rate * gradient

    if step % 100 == 0:
        print(step, loss, W)



'''
Why this matters for the Transformer

Remember:
Q = X @ Wq

During training, Wq is updated.

Likewise:
Wk
Wv
Wo
W1
W2
gamma
beta

all become learned parameters.

Conceptually:
prediction
↓
loss
↓
calculate how each parameter affected loss
↓
gradients
↓
update parameters
↓
repeat
'''


'''
Numerical gradient — useful for intuition
There's another way to estimate derivatives without deriving formulas.

def f(w):
    return w ** 2

We know mathematically:

df/dw = 2w

At: w = 3

the derivative is: 6

But numerically we can do:

eps = 1e-5

gradient = (
    f(w + eps) - f(w - eps)
) / (2 * eps)

This should also produce something very close to: 6

This is called a finite-difference gradient.
It's slow but extremely useful for checking manually derived gradients.
'''

'''
def f(w):
    return w ** 3

for w = 2

df/dw = 3(w**2)

so it is 12
'''

# Let's create something nonlinear.
# One neuron:
#
# z = XW + b
# a = ReLU(z)

X = np.array([
    [1., 2.],
    [3., 1.]
])

W = np.array([0.5, -0.2])

b = 0.1

def relu(x):
    return np.maximum(0, x)

z = X @ W + b
a = relu(z)
print(z, a)



# Main challenge

print("\n\n\nMAIN CHALLENGE\n")

np.random.seed(0)

X = np.random.randn(100, 3)

'''
The hidden true relationship will be:
y = 2*x0 - 3*x1 + 0.5*x2 + 4
'''
y_true = ( 2 * X[:, 0]
           - 3 * X[:, 1]
           + 0.5 * X[:, 2]
           + 4
           )
W = np.zeros(3)
b = 0.0

learning_rate = 0.05

for step in range(500):
    y_pred = X @ W + b

    error = y_pred - y_true
    loss = np.mean(error ** 2)

    dW = (2 / len(X)) * X.T @ error
    db = 2 * error.mean()

    W -= learning_rate * dW
    b -= learning_rate * db

    if step % 50 == 0:
        print(step, loss, W, b)