import numpy as np

def softmax(x):
    shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shift)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def split_head(x, num_heads):
    batch, token, embedding =x.shape

    if embedding % num_heads != 0:
        raise ValueError("embedding must be divisible by num_heads")

    head_dim = embedding // num_heads
    x = x.reshape(batch, token, num_heads, head_dim)
    return x.transpose(0, 2, 1, 3)

def combine_heads(x):
    batch, num_heads, token, head_dim = x.shape
    x = x.transpose(0, 2, 1, 3)
    return x.reshape(batch, token, num_heads * head_dim)

'''
Why do we need masks?
Suppose we're training a language model on:

I love machine learning

Token positions might be:
0        1        2         3
I       love    machine   learning

When the model is processing "I", it should not be allowed to look at:
love
machine
learning

because those are future tokens.
When processing "love", it can look at:
I
love

but not:
machine
learning

And when processing "machine", it can look at:
I
love
machine

but not "learning".

This is called causal attention.
The rule is:
token i may attend to token j only when:
j <= i
In other words, a token may look backward and at itself, but not forward.
'''


'''
Suppose we have four tokens.

Attention scores have shape:

(4, 4)

because every query token is compared against every key token.

Imagine:

scores = np.array([
    [2., 4., 1., 3.],
    [1., 5., 2., 0.],
    [3., 1., 4., 2.],
    [2., 3., 1., 6.]
])

Interpret the matrix as:

                KEYS

             0   1   2   3

query 0      2   4   1   3
query 1      1   5   2   0
query 2      3   1   4   2
query 3      2   3   1   6

Without masking, query 0 could pay attention to key 3.

For a GPT-like model, that's illegal because key 3 is in the future.
'''

mask = np.triu(
    np.ones((4, 4)),
    k=1
)

print(mask)

print(np.triu(np.ones((3, 3)), k=1))
# should produce
# [0, 1, 1]
# [0, 0, 1]
# [0, 0, 0]
# the diag is 0, due to k=1, because you start with a first shift

scores = np.array([
    [2., 4., 1., 3.],
    [1., 5., 2., 0.],
    [3., 1., 4., 2.],
    [2., 3., 1., 6.]
])
print(scores)
mask = np.triu(
    np.ones((4, 4), dtype=bool),
    k=1
)
print(mask)
#scores[mask] = -np.inf
masked_scores = np.where(
    mask,
    -np.inf,
    scores
)
print(scores)

weights = softmax(scores)
print(weights)
print(weights.sum(axis=-1))


mask = np.array([
    [False, True],
    [False, False]
])

scores = np.array([
    [1., 5.],
    [2., 3.]
])

masked = np.where(mask, -np.inf, scores)
print(masked)

x = np.array([
    [1, 8, 3],
    [7, 2, 9]
])
mask = np.where(5 < x, 0, x)
print(mask)

# Suppose:
# scores.shape == (32, 8, 100, 100)
#
# Interpret:
# (batch, heads, queries, keys)
#
# We need a causal mask for:
# 100 query positions × 100 key positions
#
# tokens = scores.shape[-1]
# causal_mask = np.triu( np.ones((tokens, tokens), dtype=bool),k=1)
# Shape:
# (100, 100)
# masked_scores = np.where(
#     causal_mask,
#     -np.inf,
#     scores
# )
#
# NumPy sees:
# scores:
# (32, 8, 100, 100)
#
# mask:
# (        100, 100)
#
# and broadcasts the same causal mask over:
# 32 batches
# 8 heads

# Suppose:
# scores.shape = (16, 12, 128, 128)
# and:
# causal_mask.shape = (128, 128)
#
# Explain why:
# np.where(causal_mask, -np.inf, scores)
# works.
#
# What dimensions are being broadcast?
# The dim are 16, and 12, because the two last dim of scores are the same as the mask dim

'''
Now insert masking between scaling and softmax:

tokens = Q.shape[-2]

causal_mask = np.triu(
    np.ones((tokens, tokens), dtype=bool),
    k=1
)

masked_scores = np.where(
    causal_mask,
    -np.inf,
    scaled_scores
)

weights = softmax(masked_scores)

QKᵀ
↓
scale
↓
mask
↓
softmax

'''

def causal_multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    Q, K, V = split_head(Q, num_heads), split_head(K, num_heads), split_head(V, num_heads)

    scores = Q @ K.swapaxes(-1, -2)
    scaled_scores = scores / np.sqrt(Q.shape[-1])

    tokens = Q.shape[-2]
    causal_mask = np.triu(
        np.ones((tokens, tokens), dtype=bool),
        k=1
    )
    masked_scores = np.where(causal_mask, -np.inf, scaled_scores)

    weights = softmax(masked_scores)

    heads = weights @ V

    output = combine_heads(heads)

    output = output @ Wo

    return output, weights

# Suppose:
#
# scores.shape = (32, 8, 100, 100)
# padding_mask.shape = (32, 100)
#
# What shape should you reshape the padding mask into so it broadcasts across:
#
# heads
# query positions
# into (32, 1, 1, 100)

'''
padding_mask = np.array([
    [1, 1, 1, 0],
    [1, 1, 1, 1]
])
masked_scores = np.where(
    causal_mask,
    -np.inf,
    scores
)

padding_mask = padding_mask[:, None, None, :]

masked_scores = np.where(
    padding_mask == 0,
    -np.inf,
    masked_scores
)

weights = softmax(masked_scores)

Now attention cannot use:
future tokens
or
padding tokens
'''

def causal_attention(Q, K, V):
    d = Q.shape[-1]

    scores = Q @ K.swapaxes(-1, -2)
    scores = scores / np.sqrt(d)

    tokens = Q.shape[-2]

    mask = np.triu(np.ones((tokens, tokens), dtype=bool), k=1)

    scores = np.where(mask, -np.inf, scores)

    weights = softmax(scores)

    output = weights @ V

    return output, weights

# This function works for shapes such as:
# (batch, tokens, head_dim)
#
# and:
# (batch, heads, tokens, head_dim)
# because the 2D mask broadcasts over all earlier dimensions.

np.random.seed(0)

X = np.random.randn(2, 4, 8)

num_heads = 2
embedding = 8

Wq = np.random.randn(8, 8)
Wk = np.random.randn(8, 8)
Wv = np.random.randn(8, 8)
Wo = np.random.randn(8, 8)

padding_mask = np.array([
    [1, 1, 1, 0],
    [1, 1, 1, 1]
])

def causal_multi_head_attention_challenge(X, Wq, Wk, Wv, Wo, num_heads, padding_mask=None):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    Q, K, V = split_head(Q, num_heads), split_head(K, num_heads), split_head(V, num_heads)

    scores = Q @ K.swapaxes(-1, -2)
    scores = scores / np.sqrt(Q.shape[-1])

    tokens = Q.shape[-2]
    causal_mask = np.triu(
        np.ones((tokens, tokens), dtype=bool),
        k=1
    )
    scores = np.where(causal_mask, -np.inf, scores)

    if padding_mask is not None:
        padding_mask = padding_mask[:, None, None, :]
        scores = np.where(
            padding_mask == 0,
            -np.inf,
            scores
        )

    weights = softmax(scores)

    heads = weights @ V

    output = combine_heads(heads)

    output = output @ Wo

    return output, weights

output, weights = causal_multi_head_attention_challenge(X, Wq, Wk, Wv, Wo, num_heads, padding_mask)

print(weights[0, 0])
print(weights.sum(axis=-1))