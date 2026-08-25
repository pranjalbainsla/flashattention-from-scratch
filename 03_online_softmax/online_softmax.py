# implementing online softmax for 1D 
import math

def online_softmax(x, block_size=2):
    m = -float("inf")
    l = 0.0

    for start in range(0, len(x), block_size):
        block = x[start:start + block_size]

        m_old = m
        block_max = max(block)
        m = max(m_old, block_max)

        l = (
            l * math.exp(m_old - m)
            + sum(math.exp(v - m) for v in block)
        )

    # We now have the global m and l
    # Compute the probabilities using them
    probs = [
        math.exp(v - m) / l
        for v in x
    ]

    return probs

# import torch
# x = [4., 2., 7., 3.]
# print(torch.allclose(torch.tensor(online_softmax(x)), torch.softmax(torch.tensor(x), dim=0)))

    