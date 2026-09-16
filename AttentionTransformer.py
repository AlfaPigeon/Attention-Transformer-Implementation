# **Rules** (I promise i will follow them, we can have an online meeting you can ask anything related to this implementation)
# No code generation, just code compilation (Except graph and data stuff)
# Only use ai for documentation
# Do not copy code from external sources (Except documentation examples)
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
epochs = 5

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

optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.01)

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
            test_loss += loss_fn(test_pred, y_test[i])
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f"Epoch: {epoch} | Cross Entropy Train Loss: {loss} | Cross Entropy Test Loss: {test_loss} ")


    
'''
Epoch: 0 | Cross Entropy Train Loss: 2.513230323791504 | Cross Entropy Test Loss: 2.448920249938965 
Epoch: 1 | Cross Entropy Train Loss: 2.4573280811309814 | Cross Entropy Test Loss: 2.425050973892212 
Epoch: 2 | Cross Entropy Train Loss: 2.3892598152160645 | Cross Entropy Test Loss: 2.3977952003479004 
Epoch: 3 | Cross Entropy Train Loss: 2.3094229698181152 | Cross Entropy Test Loss: 2.3692643642425537 
Epoch: 4 | Cross Entropy Train Loss: 2.2170298099517822 | Cross Entropy Test Loss: 2.341926336288452 
'''

# Visual (I generated this code with ai)

import matplotlib.pyplot as plt

# =============================================
# VISUALIZATION
# =============================================

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the training and testing lines
plt.plot(range(epochs), train_loss_values, label='Train Loss', color='blue', linewidth=2, marker='o')
plt.plot(range(epochs), test_loss_values, label='Test Loss', color='red', linewidth=2, marker='x')

# Add labels and styling
plt.title('Transformer Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Show the graph!
plt.show()


# Test

with torch.inference_mode():

    i_am = torch.tensor(
        [vocab["i"], vocab["am"]]
    )

    out = model(i_am)

    out = out.tolist()

    next_word_index = torch.argmax(torch.tensor(out[len(out)-1]))

    print("i am ", [key for key, value in vocab.items() if value == next_word_index][0])
