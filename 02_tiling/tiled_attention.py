import torch

def tiled_attention(Q, K, V, q_block_size, kv_block_size):
    """
    Compute attention block-by-block without materializing
    the full N × N attention matrix.

    Q: (N, d)
    K: (N, d)
    V: (N, d)

    Returns:
        O: (N, d)
    """

    out = torch.zeros_like(Q)

    for start_q in range(0, Q.size(0), q_block_size):

        block_Q = Q[start_q:start_q + q_block_size]  # (q_block_size, d)

        # Running statistics for each query in this Q block
        # m: (q_block_size,)   running maximum
        # l: (q_block_size,)   running softmax denominator
        # A: (q_block_size,d)  running weighted sum of V
        m = torch.full(
            (block_Q.size(0),),
            float("-inf")
        )
        l = torch.zeros(block_Q.size(0))
        A = torch.zeros_like(block_Q)
        # Refactor: use block_Q.size(0) because the final Q block may be smaller than q_block_size

        for start_kv in range(0, K.size(0), kv_block_size):
            block_K = K[start_kv:start_kv + kv_block_size]  # (kv_block_size, d)
            block_V = V[start_kv:start_kv + kv_block_size]  # (kv_block_size, d)

            scores = block_Q @ block_K.T # (q_block_size, d) @ (d, kv_block_size) -> (q_block_size, kv_block_size)

            # update running max for each query
            m_old = m
            block_max = scores.max(dim=1).values # scores.max(dim=1) returns a named tuple containing both values and indices
            m = torch.maximum(m_old, block_max) # (q_block_size,)
          

            weights = torch.exp(scores - m[:, None])  # (q_block_size, kv_block_size)

            l = (
                l * torch.exp(m_old - m)
                + weights.sum(dim=1)
            )

            A = (
                A * torch.exp(m_old - m)[:, None] # (q_block_size, d) * (q_block_size, 1)
                + (weights[:, :, None] * block_V[None, :, :]).sum(dim=1)
            )

        out[start_q: start_q + q_block_size] = A/l[:, None] # (q_block_size, d)

    return out

