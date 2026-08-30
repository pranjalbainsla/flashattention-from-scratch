import torch
import math


def flash_attention_backward(Q, K, V, dO, m, l, block_size):
    d = Q.size(-1)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # First pass: compute D = rowsum(dP * P)
    D = torch.zeros(Q.size(0), 1)

    for q_start in range(0, Q.size(0), block_size):

        Q_block = Q[q_start:q_start + block_size]
        dO_block = dO[q_start:q_start + block_size]

        m_block = m[q_start:q_start + block_size]
        l_block = l[q_start:q_start + block_size]

        D_block = torch.zeros(Q_block.size(0), 1)

        for kv_start in range(0, K.size(0), block_size):

            K_block = K[kv_start:kv_start + block_size]
            V_block = V[kv_start:kv_start + block_size]

            S_block = Q_block @ K_block.T / math.sqrt(d)

            P_block = torch.exp(
                S_block - m_block[:, None]
            ) / l_block[:, None]

            dP_block = dO_block @ V_block.T

            D_block += (
                dP_block * P_block
            ).sum(dim=-1, keepdim=True)

        D[q_start:q_start + block_size] = D_block
        
    # TODO: D can be computed directly as (O * dO).sum(dim=-1, keepdim=True),
    # eliminating this separate first pass over K/V.

    # Second pass: compute dQ, dK, dV
    for q_start in range(0, Q.size(0), block_size):

        Q_block = Q[q_start:q_start + block_size]
        dO_block = dO[q_start:q_start + block_size]

        m_block = m[q_start:q_start + block_size]
        l_block = l[q_start:q_start + block_size]
        D_block = D[q_start:q_start + block_size]

        for kv_start in range(0, K.size(0), block_size):

            K_block = K[kv_start:kv_start + block_size]
            V_block = V[kv_start:kv_start + block_size]

            S_block = Q_block @ K_block.T / math.sqrt(d)

            P_block = torch.exp(
                S_block - m_block[:, None]
            ) / l_block[:, None]

            dP_block = dO_block @ V_block.T

            dS_block = P_block * (
                dP_block - D_block
            )

            dQ[q_start:q_start + block_size] += (
                dS_block @ K_block / math.sqrt(d)
            )

            dK[kv_start:kv_start + block_size] += (
                dS_block.T @ Q_block / math.sqrt(d)
            )

            dV[kv_start:kv_start + block_size] += (
                P_block.T @ dO_block
            )

    return dQ, dK, dV