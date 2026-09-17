# Implementation Of Attention Transformer



### Scaled Dot-Product Attention

This function computes the attention scores by taking the dot product of the Queries and Keys, scaling them by the square root of the dimension, and applying a softmax function before multiplying by the Values.

```python
def ScaledDotProductAttention(Q: NDArray, K: NDArray, V: NDArray, d: float) -> NDArray:
    output = Q @ K.T
    output = output / np.sqrt(d) # Attention Scores
    output = softmax(output)
    output = output @ V

    return output
```


## References

* Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2023). **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**. *arXiv preprint arXiv:1706.03762*.