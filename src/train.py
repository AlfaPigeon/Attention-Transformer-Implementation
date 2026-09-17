import torch
from torch import nn, softmax
import numpy as np
from numpy.typing import NDArray
from data_setup import vocab, x_train, y_train, x_test, y_test
from model import AttentionTransformer
import matplotlib.pyplot as plt

from utils import test_model, visualize_loss

# Paramerters ==========
embed_dim = 8
epochs = 10

torch.manual_seed(238972198372)
# ======================

train_loss_values = []
test_loss_values = []

model = AttentionTransformer(embed_dim=embed_dim, vocab=vocab)

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
Before Positional Embeddings, Single-Head Attention was used.

Epoch: 0 | Cross Entropy Train Loss: 2.5329372882843018 | Cross Entropy Test Loss: 2.651475667953491 
Epoch: 1 | Cross Entropy Train Loss: 2.475619316101074 | Cross Entropy Test Loss: 2.64501953125 
Epoch: 2 | Cross Entropy Train Loss: 2.414933919906616 | Cross Entropy Test Loss: 2.6390790939331055 
Epoch: 3 | Cross Entropy Train Loss: 2.3519277572631836 | Cross Entropy Test Loss: 2.631481885910034 
Epoch: 4 | Cross Entropy Train Loss: 2.283769130706787 | Cross Entropy Test Loss: 2.6199395656585693 
Epoch: 5 | Cross Entropy Train Loss: 2.2080681324005127 | Cross Entropy Test Loss: 2.6021711826324463 
Epoch: 6 | Cross Entropy Train Loss: 2.123443603515625 | Cross Entropy Test Loss: 2.5765445232391357 
Epoch: 7 | Cross Entropy Train Loss: 2.0292766094207764 | Cross Entropy Test Loss: 2.542869806289673 
Epoch: 8 | Cross Entropy Train Loss: 1.925527572631836 | Cross Entropy Test Loss: 2.50300669670105 
Epoch: 9 | Cross Entropy Train Loss: 1.8126715421676636 | Cross Entropy Test Loss: 2.461205244064331 

After Positional Embeddings, Single-Head Attention was used.

Epoch: 0 | Cross Entropy Train Loss: 2.5703442096710205 | Cross Entropy Test Loss: 2.8255271911621094 
Epoch: 1 | Cross Entropy Train Loss: 2.5186352729797363 | Cross Entropy Test Loss: 2.794820785522461 
Epoch: 2 | Cross Entropy Train Loss: 2.4640650749206543 | Cross Entropy Test Loss: 2.7617530822753906 
Epoch: 3 | Cross Entropy Train Loss: 2.4118716716766357 | Cross Entropy Test Loss: 2.725377082824707 
Epoch: 4 | Cross Entropy Train Loss: 2.3612186908721924 | Cross Entropy Test Loss: 2.6853458881378174 
Epoch: 5 | Cross Entropy Train Loss: 2.309675455093384 | Cross Entropy Test Loss: 2.6418468952178955 
Epoch: 6 | Cross Entropy Train Loss: 2.2542884349823 | Cross Entropy Test Loss: 2.595872163772583 
Epoch: 7 | Cross Entropy Train Loss: 2.1917102336883545 | Cross Entropy Test Loss: 2.5487964153289795 
Epoch: 8 | Cross Entropy Train Loss: 2.118272066116333 | Cross Entropy Test Loss: 2.501823902130127 
Epoch: 9 | Cross Entropy Train Loss: 2.03072190284729 | Cross Entropy Test Loss: 2.4565365314483643 

'''

visualize_loss(epochs, train_loss_values, test_loss_values)
test_model(model, vocab)