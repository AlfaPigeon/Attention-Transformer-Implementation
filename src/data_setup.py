# Vocabulary
import torch

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

# ===========================