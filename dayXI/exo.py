import torch
import torch.nn as nn
import numpy as np

x = np.array([
    [1., 2.],
    [3., 4.]
])

print(x)
print(x.shape)

x = torch.tensor([
    [1., 2.],
    [3., 4.]
])

print(x)
print(x.shape)

X = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.]
])

print(X.shape)
print(X.mean())
print(X.mean(dim=0))
print(X.mean(dim=1))
print(X.T.shape)

w = torch.tensor(3.0, requires_grad=True)
y = w ** 2
y.backward()
print(w.grad)

w = torch.tensor(2.0, requires_grad=True)
y = w**3
y.backward()
print(w.grad) # should be 12

x = torch.tensor(2.0, requires_grad=True)

a = 3 * x
b = a ** 2
L = 5 * b

L.backward()
print(x.grad)


'''
y = 2*x0 - 3*x1 + 0.5*x2 + 4
'''

torch.manual_seed(0)
X = torch.randn(100, 3)

Y_true = (2 * X[:, 0]
          - 3 * X[:, 1]
          + 0.5 * X[:, 2]
          + 4)

W = torch.zeros(3, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

#Y_pred = X @ W + b
#loss = ((Y_pred - Y_true) * 2).mean()

#loss.backward()
#print(W.grad)
#print(b.grad)

#learning_rate = 0.05

#with torch.no_grad():
#    W -= learning_rate * W.grad
#    b -= learning_rate * b.grad

'''
What's:
torch.no_grad()
doing?

It tells PyTorch:
Don't build a computation graph for these operations.

We don't want:
W -= learning_rate * gradient

to itself become part of the model's mathematical computation graph.

Then clear gradients:
W.grad.zero_()
b.grad.zero_()
'''

learning_rate = 0.05

for step in range(500):
    Y_pred = X @ W + b

    loss = ((Y_pred - Y_true) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    W.grad.zero_()
    b.grad.zero_()

    if step % 50 == 0:
        print(step, loss.item(), W, b)

# manual layer
# Z = X @ W + b

layer = nn.Linear(
    in_features=3,
    out_features=8
)
# if X.shape == (32, 3)
# then Y = layer(X) has (32, 8)
# The layer owns its own:
# weights
# bias
#
# You can inspect them:
# print(layer.weight.shape)
# print(layer.bias.shape)


'''
layer = nn.Linear(10, 64)

Predict:
layer.weight.shape = (64, 10)
layer.bias.shape = (64,)

Then:
X = torch.randn(32, 10)
Y = layer(X)

What is:

Y.shape = (32, 64)
'''
layer = nn.Linear(10, 64)
print(layer.bias.shape)
X = torch.randn(32, 10)
Y = layer(X)
print(Y.shape)



model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

X = torch.randn(200, 2)

Y_pred = model(X)

print(Y_pred.shape)

print(model)
for name, parameter in model.named_parameters():
    print(name, parameter.shape)


#loss = ((Y_pred - Y_true) ** 2).mean()
#loss_fn = nn.MSELoss()
#loss = loss_fn(Y_pred, Y_true)

#optimizer = torch.optim.SGD(
#    model.parameters(),
#    lr=0.01
#)  # instead of W1 -= ..., .....

# Then training loop become

#optimizer.zero_grad()

#Y_pred = model(X)

#loss = loss_fn(Y_pred, Y_true)

#loss.backward()

#optimizer.step()
# The optimizer finds all model parameters and applies their gradients.


torch.manual_seed(0)
X = torch.empty(200, 2).uniform_(-2, 2)
Y_true = (
    X[:, 0] ** 2
    + 2 * X[:, 1]
).reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

for step in range(5000):
    optimizer.zero_grad()

    Y_pred = model(X)

    loss = loss_fn(Y_pred, Y_true)

    loss.backward()

    if step == 0:
        for name, parameter in model.named_parameters():
            print(name)
            print("parameter:", parameter.shape)
            print("gradient:", parameter.grad.shape)

    optimizer.step()

    if step % 500 == 0:
        print(step, loss.item())


test = torch.tensor([
    [2., 1.],
    [1., 1.5],
    [-2., 0.]
])
with torch.no_grad():
    prediction = model(test)
print(prediction)


class TinyNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

model = TinyNetwork()
prediction = model(X)


# y = x0² + 2*x1

torch.manual_seed(0)

# -------------------------
# DATA
# -------------------------

X = torch.empty(1000, 2).uniform_(-2, 2)

Y_true = (X[:, 0] ** 2 + 2 * X[:, 1]).reshape(-1, 1)

# -------------------------
# MODEL
# -------------------------

model = TinyNetwork()

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

# -------------------------
# TRAIN
# -------------------------

model.train()

for step in range(5000):

    # Remove previous gradients
    optimizer.zero_grad()

    # Forward pass
    Y_pred = model(X)

    # Calculate error
    loss = loss_fn(Y_pred, Y_true)

    # Backpropagation
    loss.backward()

    # Optional: inspect gradients once
    if step == 0:
        for name, parameter in model.named_parameters():
            print(
                name,
                "parameter:",
                parameter.shape,
                "gradient:",
                parameter.grad.shape
            )

    # Update parameters
    optimizer.step()

    if step % 500 == 0:
        print(step, loss.item())

# -------------------------
# TEST
# -------------------------


test = torch.tensor([
    [ 1.5,  1.0],
    [ 1.0,  1.5],
    [-1.5,  0.5],
    [ 0.0, -1.0]
])

model.eval()

with torch.no_grad():
    predictions = model(test)

print(predictions)