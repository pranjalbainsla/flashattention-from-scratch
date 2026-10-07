# Why is attention expensive?

Attention outputs an N × d matrix, but the standard computation materializes N × N score and probability matrices. Writing these N × N intermediates to and from GPU memory (HBM) is what makes standard attention slow and memory-hungry, and that makes us think:

> **Can we compute O = softmax(QKᵀ)V without ever materializing the full N × N matrices?**  

For a particular query row i:

$$
O_i = \sum_j P_{ij} V_j
$$

and:

$$
P_{ij} = \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})}
$$

Therefore:

$$
O_i = \frac{\sum_j \exp(S_{ij}) V_j}
{\sum_j \exp(S_{ij})}
$$

This tells us something important:

To produce one output row, we do not fundamentally need to store the entire probability row.

We only need enough information to compute:

1. the normalization factor
2. the weighted sum of V

With numerical stabilization, we can maintain a **running maximum** and **running normalization factor**.

This observation will eventually allow us to process attention in blocks instead of materializing the entire N × N matrices.

