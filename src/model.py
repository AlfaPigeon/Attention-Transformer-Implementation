import torch
from torch import nn

class AttentionTransformer(nn.Module):

    def __init__(self, embed_dim, vocab):
        super().__init__()

        self.d_k = embed_dim
        self.embed_dim = embed_dim
        self.vocab = vocab

        self.embedding  = nn.Embedding(
            num_embeddings=len(self.vocab),
            embedding_dim=self.embed_dim
        )

        self.QW = nn.Linear(self.embed_dim, self.embed_dim)
        self.KW = nn.Linear(self.embed_dim, self.embed_dim)
        self.VW = nn.Linear(self.embed_dim, self.embed_dim)

        self.output = nn.Linear(
            self.embed_dim,
            len(self.vocab)
        )

    def positional(self, x):

        positions = torch.arange(0, x.size(0), dtype=torch.float)
        positions = positions.unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.embed_dim)
        )

        pe = torch.zeros(x.size(0), self.embed_dim)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        return pe

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

        word_embeddings = self.embedding(x)
        positional_embeddings = self.positional(x)
        x = word_embeddings + positional_embeddings
        x = self.attention(x) # Altered Embedding Vectors
        x = self.output(x)

        return x
