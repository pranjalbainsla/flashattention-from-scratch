import torch

def flash_attention(Q, K, V, q_block_size, kv_block_size, causal=False):
    """
    Reference implementation of FlashAttention.

    Computes:
        softmax(QK^T / sqrt(d)) V

    without materializing the full N x N attention matrix.

    Q: (N, d)
    K: (N, d)
    V: (N, d)

    Returns:
        O: (N, d)
    """

    N, d = Q.shape
    out = torch.zeros_like(Q)

    scale = 1.0 / (d ** 0.5) # to keep the magnitude/variance of QKᵀ scores controlled as d grows

    for start_q in range(0, N, q_block_size):

        block_Q = Q[start_q:start_q + q_block_size]
        Bq = block_Q.size(0)

        m = torch.full(
            (Bq,),
            float("-inf"),
            device=Q.device,
            dtype=Q.dtype,
        )

        l = torch.zeros(
            Bq,
            device=Q.device,
            dtype=Q.dtype,
        )

        A = torch.zeros_like(block_Q)

        for start_kv in range(0, N, kv_block_size):

            block_K = K[start_kv:start_kv + kv_block_size]
            block_V = V[start_kv:start_kv + kv_block_size]

            scores = block_Q @ block_K.T
            scores = scores * scale

            Bk = block_K.size(0)

            if causal:
                q_positions = torch.arange(
                    start_q,
                    start_q + Bq,
                    device=Q.device,
                )

                kv_positions = torch.arange(
                    start_kv,
                    start_kv + Bk,
                    device=Q.device,
                )

                mask = q_positions[:, None] < kv_positions[None, :]
                scores = scores.masked_fill(mask, float("-inf"))

            m_old = m

            block_max = scores.max(dim=1).values
            m = torch.maximum(m_old, block_max)

            weights = torch.exp(scores - m[:, None])

            l = (
                l * torch.exp(m_old - m)
                + weights.sum(dim=1)
            )

            A = (
                A * torch.exp(m_old - m)[:, None]
                + (weights[:, :, None] * block_V[None, :, :]).sum(dim=1)
            )

        out[start_q:start_q + Bq] = A / l[:, None]

    return out