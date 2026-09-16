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

        self.QW = nn.Linear(embed_dim, embed_dim)
        self.KW = nn.Linear(embed_dim, embed_dim)
        self.VW = nn.Linear(embed_dim, embed_dim)

        self.layer1 = nn.Linear(
            embed_dim,
            16
        )

        self.output = nn.Linear(
            16,
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



train_loss_values = []
test_loss_values = []
# =============================================

model = AttentionTransformer()

optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.03)

loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):

    model.train()

    for i in range(len(x_train)):

        y_pred = model(x_train[i])

        loss = loss_fn(y_pred, y_train[i])

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    # Evaluate Model

    model.eval()

    with torch.inference_mode():
        test_loss = 0
        for i in range(len(x_test)):
            test_pred = model(x_test[i])
            test_loss += loss_fn(test_pred, y_test[i]) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")


    
