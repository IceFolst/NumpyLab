import numpy as np

from day7.exo import causal_multi_head_attention_challenge


def softmax(x):
    shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shift)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def split_head(x, num_heads):
    batch, tokens, embedding = x.shape
    heads_dim = embedding // num_heads
    x = x.reshape(batch, tokens, num_heads, heads_dim)
    return x.transpose(0, 2, 1, 3)

def combine_head(x):
    batch, num_heads, tokens, heads_dim = x.shape
    x = x.transpose(0, 2, 1, 3)
    return x.reshape(batch, tokens, num_heads * heads_dim)

def causal_attention(Q, K, V):
    d = Q.shape[-1]

    scores = Q @ K.swapaxes(-1, -2)
    scores = scores / np.sqrt(d)

    tokens = Q.shape[-2]
    mask = np.triu(
        np.ones((tokens, tokens), dtype=bool), k=1
    )
    scores = np.where(mask, -np.inf, scores)

    weights = softmax(scores)

    output = weights @ V

    return output, weights

'''
Residual connections

Suppose:
X.shape == (32, 100, 512)

and attention produces:
attention_output.shape == (32, 100, 512)

A residual connection simply does:
X = X + attention_output

The idea is:

original information
        +
transformed information

Instead of replacing X completely.

For example:

x = np.array([1., 2., 3.])
f_x = np.array([0.5, -0.2, 1.])

output = x + f_x

gives:

[1.5, 1.8, 4.0]
'''

X = np.array([
    [1., 2., 3.],
    [4., 5., 6.]
])

F = np.array([
    [0.5, 1., -1.],
    [2., -2., 0.5]
])

Y = X + F
print(Y)

# X.shape = (32, 100, 512)
# F.shape = (32, 100, 256) would fail, because the lasts dim are not the same,
# or 1 and x, or other way to broadcast

# Layer normalization
#
# Now we introduce LayerNorm.
#
# Suppose one token embedding is:
#
x = np.array([2., 4., 6., 8.])

#LayerNorm takes the values inside that token and normalizes them.
#
#Conceptually:
#             x - mean
#normalized = --------
#               std

mean = np.mean(x)
std = np.std(x)
normalized = (x - mean) / std
print(normalized.mean())
print(normalized.std())

X = np.random.randn(32, 100, 512)
mean = X.mean(axis=-1, keepdims=True)
std = X.std(axis=-1, keepdims=True)

X_norm = (X - mean) / std
print(mean.shape)


x = np.array([5., 5., 5., 5.])
eps = 1e-5
normalized = (x - mean) / np.sqrt(x.var() + eps)


def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)

    return (x - mean) / np.sqrt(var + eps)

X = np.random.randn(2, 3, 4)
Y = layer_norm(X)
print(Y.shape)
print(Y.mean(axis=-1))
print(Y.var(axis=-1))

'''
Actual LayerNorm does a little more.
After normalization, it applies:

output = gamma * normalized + beta

where:
gamma = scale
beta  = shift

Both are learned.

For an embedding dimension of 512:
gamma = np.ones(512)
beta = np.zeros(512)

Then:
output = gamma * normalized + beta
'''

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)

    normalized = (x - mean) / np.sqrt(var + eps)

    return gamma * normalized + beta


'''
Attention lets tokens exchange information.
The Transformer also contains a small neural network applied independently to each token.
It's often called:
FFN
or:
MLP

A simple feed-forward network is:
x
↓
linear
↓
activation
↓
linear
↓
output

Mathematically:
FFN(x) = activation(xW1 + b1) W2 + b2
For today we'll use ReLU:
'''

def relu(x):
    return np.maximum(0, x)

def feed_forward(x, w1, b1, w2, b2):
    hidden = x @ w1 + b1
    hidden = relu(hidden)

    return hidden @ w2 + b2

'''
Why does the FFN expand the dimension?

Suppose:
embedding = 512

A common pattern is to expand to a larger hidden dimension:
512 → 2048 → 512

So:
W1.shape == (512, 2048)
b1.shape == (2048,)

W2.shape == (2048, 512)
b2.shape == (512,)

Given:
X.shape = (32, 100, 512)

first layer:
(32, 100, 512)
@
          (512, 2048)
→ (32, 100, 2048)

then ReLU doesn't change the shape:
(32, 100, 2048)

then:
(32, 100, 2048)
@
          (2048, 512)
→ (32, 100, 512)

So the FFN returns to the original embedding dimension.
That's important because we want another residual connection:
X + ffn_output
'''

# In equations:
#
# X1 = X + attention(layer_norm(X))
#
# output = X1 + feed_forward(layer_norm(X1))



def transformer_block(X, Wq, Wk, Wv, Wo, W1, b1, W2, b2,
    gamma1, beta1, gamma2, beta2, num_heads,
    padding_mask=None
):
    norm1 = layer_norm(X, gamma1, beta1)

    attention_output, weights = causal_multi_head_attention_challenge(norm1, Wq, Wk, Wv,
                                                                      Wo, num_heads, padding_mask)

    X = X + attention_output

    norm2 = layer_norm(X, gamma2, beta2)

    ffn_output = feed_forward(norm2, W1, b1, W2, b2)

    X = X + ffn_output

    print("X:", X.shape)
    print("norm1:", norm1.shape)
    print("attention:", attention_output.shape)
    print("norm2:", norm2.shape)
    print("ffn:", ffn_output.shape)

    return X, weights




np.random.seed(0)

batch = 2
tokens = 4
embedding = 8
num_heads = 2
hidden = 16

X = np.random.randn(batch, tokens, embedding)

Wq = np.random.randn(embedding, embedding)
Wk = np.random.randn(embedding, embedding)
Wv = np.random.randn(embedding, embedding)
Wo = np.random.randn(embedding, embedding)

W1 = np.random.randn(embedding, hidden)
b1 = np.random.randn(hidden)

W2 = np.random.randn(hidden, embedding)
b2 = np.random.randn(embedding)

gamma1 = np.ones(embedding)
beta1 = np.zeros(embedding)

gamma2 = np.ones(embedding)
beta2 = np.zeros(embedding)

padding_mask = np.array([
    [1, 1, 1, 0],
    [1, 1, 1, 1]
])

output, weights = transformer_block(X, Wq, Wk, Wv, Wo, W1, b1, W2, b2,
    gamma1, beta1, gamma2, beta2, num_heads, padding_mask)

test = layer_norm(
    X,
    np.ones(embedding),
    np.zeros(embedding)
)

print(test.mean(axis=-1))
print(test.var(axis=-1))


'''
Transformer block shape flow

This is worth being able to reproduce from memory:

X
(32, 100, 512)

↓ LayerNorm

(32, 100, 512)

↓ Q/K/V projections

(32, 100, 512)

↓ split 8 heads

(32, 8, 100, 64)

↓ QKᵀ

(32, 8, 100, 100)

↓ mask + softmax

(32, 8, 100, 100)

↓ weights @ V

(32, 8, 100, 64)

↓ combine heads

(32, 100, 512)

↓ output projection

(32, 100, 512)

↓ residual

(32, 100, 512)

↓ LayerNorm

(32, 100, 512)

↓ FFN first layer

(32, 100, 2048)

↓ activation

(32, 100, 2048)

↓ FFN second layer

(32, 100, 512)

↓ residual

(32, 100, 512)

If you understand that flow rather than memorizing it, 
you're developing the exact tensor intuition I wanted this bootcamp to build.
'''


## For bonus 1

## the hidden ffn shape[-1] is not the same as the X.shape[-1] so you cannot do the broadcast, due to the dimension
## for ffn_output shape, the last dim is the same so the broadcast can works

## For the bonus 2

## X = np.random.randn(32, 100, 512)

# X.mean(axis=0)
# X.mean(axis=1)
# X.mean(axis=2)
# X.mean(axis=-1)

# For the bonus 3

# Suppose:
# embedding = 512
# hidden = 2048

