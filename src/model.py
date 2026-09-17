import torch
from torch import nn

class AttentionTransformer(nn.Module):

    def __init__(self, num_heads, embed_dim, vocab):
        super().__init__()

        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.vocab = vocab

        self.embedding  = nn.Embedding(
            num_embeddings=len(self.vocab),
            embedding_dim=self.embed_dim
        )

        self.attention_heads_Q = nn.ModuleList([
            nn.Linear(self.embed_dim, self.d_k)
            for _ in range(self.num_heads)
        ])

        self.attention_heads_K = nn.ModuleList([
            nn.Linear(self.embed_dim, self.d_k)
            for _ in range(self.num_heads)
        ])

        self.attention_heads_V = nn.ModuleList([
            nn.Linear(self.embed_dim, self.d_k)
            for _ in range(self.num_heads)
        ])

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


        Q_list = []
        K_list = []
        V_list = []

        attention_list = []

        for i in range(self.num_heads):

            Q_list.append(self.attention_heads_Q[i](x))
            K_list.append(self.attention_heads_K[i](x))
            V_list.append(self.attention_heads_V[i](x))

            score = Q_list[i] @ K_list[i].T

            softmax_score = torch.softmax(
                score / torch.sqrt(
                    torch.tensor(
                        self.d_k,
                        dtype=Q_list[i].dtype
                    )
                ),
                dim=-1
            )

            attention_list.append(softmax_score)

        output_list = [
            attention_list[i] @ V_list[i]
            for i in range(self.num_heads)
        ]

        return torch.concat(output_list, dim=-1)


    def forward(self, x):

        word_embeddings = self.embedding(x)
        positional_embeddings = self.positional(x)
        x = word_embeddings + positional_embeddings
        x = self.attention(x)
        x = self.output(x)

        return x
