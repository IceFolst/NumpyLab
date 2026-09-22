import numpy as np


def softmax(x):
    shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shift)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Start with an input tensor

X = np.random.randn(2, 4, 8)
# (batch, tokens, embedding)
# (2, 4, 8)

# So:
# 2 sequences
# 4 tokens per sequence
# 8 numbers per token
#
# One token:    X[0, 0]   has shape:  (8,)
# One sequence: X[0]        has shape:  (4, 8)

print(X[0].shape)  # shape = (4, 8)
print(X[:, 0, :].shape) # shape = (2, 8)
print(X[0, 2].shape) # shape = (8,)
print(X[:, :, 3].shape) # shape = (2, 4)


# We create three weight matrices:
Wq = np.random.randn(8, 8)
Wk = np.random.randn(8, 8)
Wv = np.random.randn(8, 8)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

# X  = (2, 4, 8)
# Wq =       (8, 8)
# Q  = (2, 4, 8)

# X.shape  = (32, 100, 512)
# Wq.shape = (512, 512)
# Wk.shape = (512, 512)
# Wv.shape = (512, 512)

# Q.shape = (32, 100, 512)
# K.shape = (32, 100, 512)
# V.shape = (32, 100, 512)
# Then suppose:  Wq.shape = (512, 256)
# Q.shape would become (32, 100, 256)

# Even though all three come from X, the learned projections allow the model to represent different roles.
#
# Conceptually:
#
# Q = What am I looking for?
# K = What information do I contain?
# V = What information should I contribute?
#
# Don't take that too literally, but it’s a useful intuition.

'''
Suppose:
embedding = 8
num_heads = 2

Then:
head_dim = 8 / 2 = 4

Initially:
Q.shape = (2, 4, 8)

We reshape:
Q = Q.reshape(2, 4, 2, 4)

Now:
(batch, tokens, heads, head_dim)
(2, 4, 2, 4)

But attention is easier with:
(batch, heads, tokens, head_dim)

So:
Q = Q.transpose(0, 2, 1, 3)

Now:
(2, 2, 4, 4)

Do the same for K and V.
'''

#batch, tokens, embedding = X.shape
#num_heads = 2
#head_dim = embedding // num_heads
#Q = Q.reshape(batch, tokens, num_heads, head_dim)
#Q = Q.transpose(0, 2, 1, 3)


# X.shape = (32, 100, 512)
# num_heads = 8
# head_dim = 64
# Q.reshape(32, 100, 8, -1) shape = (32, 100, 8, 64)
# .transpose(0, 2, 1, 3) shape = (32, 8, 100, 64)

def split_head(x, num_heads):
    batch, tokens, embedding = x.shape

    if embedding % num_heads != 0:
        raise ValueError("embedding dimension must be divisible by num_heads")

    head_dim = embedding // num_heads

    x = x.reshape(batch, tokens, num_heads, head_dim)
    x = x.transpose(0, 2, 1, 3)

    return x

# embedding=512, heads=8 good => 64
# embedding=768, heads=12 good => 64
# embedding=256, heads=6  not good
# embedding=1024, heads=16  good => 64
# embedding=100, heads=3 not good


# Q.shape = (16, 4, 50, 32)
# K.shape = (16, 4, 50, 32)
# K.swapaxes(-1, -2).shape = (16, 4, 32, 50)
# scores.shape = (16, 4, 50, 50)
# weights.shape = (16, 4, 50, 50) and divided by d= 32


# head_output = weights @ V
# Suppose:
#
# weights.shape = (32, 8, 100, 100)
# V.shape       = (32, 8, 100, 64)
# output = (32, 8, 100, 64)
# So each head produces:    64 features per token

# Currently:
#
# head_output.shape =  (batch, heads, tokens, head_dim)

# Example:
# (32, 8, 100, 64)
#
# But we want:
# (batch, tokens, embedding)
# (32, 100, 512)
#
# First transpose:
# output = head_output.transpose(0, 2, 1, 3)
#
# Shape:
# (32, 100, 8, 64)
#
# Then reshape:
# output = output.reshape(32, 100, 512)
#
# because: 8 × 64 = 512
# We've recombined the heads.

def combine_heads(x):
    batch, heads, tokens, head_dim = x.shape

    x = x.transpose(0, 2, 1, 3)
    x = x.reshape(batch, tokens, heads * head_dim)
    return x

# x.shape = (64, 8, 128, 64)
# x.transpose(0, 2, 1, 3) => shape = (64, 128, 8, 64)
# after head combine => shape = (64, 128, 514

X = np.random.randn(2, 4, 8)

num_heads = 2
embedding = 8

Wq = np.random.randn(embedding, embedding)
Wk = np.random.randn(embedding, embedding)
Wv = np.random.randn(embedding, embedding)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

Q = split_head(Q, num_heads)
K = split_head(K, num_heads)
V = split_head(V, num_heads)

scores = Q @ K.swapaxes(-1, -2)

scores = scores / np.sqrt(Q.shape[-1])

weights = softmax(scores)

head_output = weights @ V

output = combine_heads(head_output)

# Final shape:  (2, 4, 8)
#
# Same shape as X.


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv

    Q, K, V = split_head(Q, num_heads), split_head(K, num_heads), split_head(V, num_heads)

    score = Q @ K.swapaxes(-1, -2)
    scaled_scores = score / np.sqrt(Q.shape[-1])

    weights = softmax(scaled_scores)

    heads = weights @ V

    output = combine_heads(heads)

    output = output @ Wo

    return output

# self-attention,
# because Q, K, V all come from the same X.
# So one sequence is attending to itself.
# Later, in cross-attention, Q could come from one source and K/V from another.

# Main challenge

np.random.seed(0)

X = np.random.randn(2, 3, 4)

num_heads = 2
embedding = 4

Wq = np.random.randn(4, 4)
Wk = np.random.randn(4, 4)
Wv = np.random.randn(4, 4)
Wo = np.random.randn(4, 4)

Q, K, V = X @ Wq, X @ Wk, X @ Wv
Q, K, V = split_head(Q, num_heads), split_head(K, num_heads), split_head(V, num_heads)

scores = Q @ K.swapaxes(-1, -2)
scaled_scores = scores / np.sqrt(Q.shape[-1])

weights = softmax(scaled_scores)
print(weights.sum(axis=-1))

heads = weights @ V

output = combine_heads(heads)

output = output @ Wo

def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv

    Q, K, V = split_head(Q, num_heads), split_head(K, num_heads), split_head(V, num_heads)

    score = Q @ K.swapaxes(-1, -2)
    scaled_scores = score / np.sqrt(Q.shape[-1])

    weights = softmax(scaled_scores)

    heads = weights @ V

    output = combine_heads(heads)

    output = output @ Wo

    return output, weights