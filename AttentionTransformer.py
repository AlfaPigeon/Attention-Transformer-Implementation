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
epochs = 10

torch.manual_seed(10)

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

def ScaledDotProductAttention(Q: NDArray, K: NDArray, V: NDArray, d: float) -> NDArray:
    output = Q @ K.T
    output = output/np.sqrt(d) # Attention Scores
    output = softmax(output)
    output = output @ V

    return output
    
class AttentionTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.d_k = embed_dim

        self.embedding  = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=embed_dim
        )

        self.QW = nn.Linear(embed_dim, embed_dim, dtype=torch.float32)
        self.KW = nn.Linear(embed_dim, embed_dim, dtype=torch.float32)
        self.VW = nn.Linear(embed_dim, embed_dim, dtype=torch.float32)

        self.layer1 = self.output = nn.Linear(
            embed_dim,
            len(vocab)
        )

        self.output = nn.Linear(
            embed_dim,
            len(vocab)
        )

    def attention(self, x):

        Q = self.QW(x)
        K = self.KW(x)
        V = self.VW(x)

        scores = Q @ K.T

        scores = scores / torch.sqrt(
            torch.tensor(
                self.d_k,
                dtype=Q.dtype
            )
        )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        return attention @ V


    def forward(self, x):

        x = self.embedding(x)
        x = self.attention(x) # Altered Embedding Vectors
        x = self.layer1(x) # Single Linear Layer
        x = torch.relu(x)
        x = self.output(x)

        return x


# TRAINING DATA (I generated this with ai) ===
# Sentences: 
# 1. "i am very cool"
# 2. "i have a lot of testosterone"
# 3. "i am masculine and superior"

x_train = [
    torch.tensor([1, 2, 3]),
    torch.tensor([1, 8, 12, 10, 11]),
    torch.tensor([1, 2, 6, 5])
]

y_train = [
    torch.tensor([2, 3, 4]),
    torch.tensor([8, 12, 10, 11, 7]),
    torch.tensor([2, 6, 5, 9])
]

# TEST DATA
# Sentence: "i have superior testosterone"

x_test = [torch.tensor([1, 8, 9])]
y_test = [torch.tensor([8, 9, 7])]

# =============================================

model = AttentionTransformer()

optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.03)

loss_fn = nn.L1Loss()


for epoch in range(epochs):

    model.train()

    pred = model.forward()

    loss = loss_fn()

    loss.bacward()

    optimizer.step()

    optimizer.zero_grad()
