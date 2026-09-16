# **Rules** (I promise i will follow them, we can have an online meeting you can ask anything related to this implementation)
# No code generation, just code compilation
# Only use ai for documentation
# Do not copy code from external sources
# Will not look at direct implementation of attention transformer
# Arms and legs tied up, face on the floor
# Will not use jupiter notebook it is wierd

# **Strategy** (What is in my mind when starting)
# Implement what I understand first
# Implement word embedding
# Implement attention layer
# Implement architecture draft
# Test and tinker until it works xd

#===========================================================================================================================

import torch
from torch import nn, softmax
import numpy as np
from numpy.typing import NDArray


# Paramerters
embed_dim = 8


# Vocabulary
vocab = {
    "<pad>": 0,
    "i": 1,
    "am": 2,
    "very": 3,
    "cool": 4,
    "and": 5,
    "masculine":6,
    "testosterone":7,
    "have": 8,
    "superior":9,
    "lot": 10,
    "of": 11,
    "a":12
}

sentence = "i have a lot of testosterone"

# Tokenization
tokens = sentence.lower().split()

token_ids = [vocab[token] for token in tokens]

x = torch.tensor(token_ids)

print(token_ids)

embedding = nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=embed_dim
)

vectors = embedding(x)

print(vectors.shape)


def ScaledDotProductAttention(Q: NDArray, K: NDArray, V: NDArray, d: float) -> NDArray:
    output = Q @ K.T
    output = output/np.sqrt(d)
    output = softmax(output)
    output = output @ V

    return output
    
