import torch

# ---------------- first draft --------------------
# def O_online(Q, K, V, block_size):
#   # Q is 1-D rn
#   m = -float("inf")
#   l = 0.0
#   A = torch.zeros_like(V[0])

#   for start in range(0, K.size(0), block_size):
#     block_K = K[start:start + block_size]
#     block_V = V[start:start + block_size]

#     m_old = m
#     scores = Q @ block_K.T
#     block_max = max(scores)
#     m = max(m_old, block_max)
      
#     l = (
#         l * math.exp(m_old - m)
#         + sum(math.exp(score - m) for score in scores)
#     )

#     weights = torch.exp(scores - m)

#     A = (
#         A * torch.exp(m_old - m)
#         + (weights[:, None] * block_V).sum(dim=0)
#     )
    
#     O = A/l

#   return O


def online_attention(Q, K, V, block_size):
    """
    Compute attention for a single query without materializing
    the full attention score/probability vector.

    Q: (d,)
    K: (N, d)
    V: (N, d)

    Returns:
        O: (d,)
    """

    # ------------ Running softmax statistics --------------
    # m = maximum score seen so far
    # l = sum of exp(score - m) over all scores seen so far
    m = torch.tensor(float("-inf"))
    l = torch.tensor(0.0)

    # ---------------- Running numerator ------------------
    # A = sum(exp(score - m) * V) over all scores seen so far
    A = torch.zeros_like(V[0])

    for start in range(0, K.size(0), block_size):
        block_K = K[start:start + block_size]  # (block_size, d)
        block_V = V[start:start + block_size]  # (block_size, d)

        # Compute scores only for the current K block.
        scores = Q @ block_K.T # (d,) @ (d, block_size) -> (block_size,)

        # The global maximum may change when we see this block
        m_old = m
        block_max = scores.max()
        m = torch.maximum(m_old, block_max)

        weights = torch.exp(scores - m)  # (block_size,)

        l = (
            l * torch.exp(m_old - m)
            + weights.sum()
        )

        A = (
            A * torch.exp(m_old - m)
            + (weights[:, None] * block_V).sum(dim=0) # Weight each V vector by its attention weight, then sum
        )

    # Normalize the accumulated numerator
    return A / l