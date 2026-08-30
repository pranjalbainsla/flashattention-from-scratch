import torch
import math

torch.manual_seed(42)

N = 8
d = 4
block_size = 2

Q = torch.randn(N, d)
K = torch.randn(N, d)
V = torch.randn(N, d)

dO = torch.randn(N, d)

# -----------------------------------------------------------
# 1. Forward pass
#    Compute the quantities used by the backward pass.

S = Q @ K.T / math.sqrt(d)
P = torch.softmax(S, dim=-1)

# FlashAttention forward saves these statistics so that P
# can be reconstructed later without storing the full matrix.

m = S.max(dim=-1).values
l = torch.exp(S - m[:, None]).sum(dim=-1)

# -----------------------------------------------------------
# 2. Naive backward pass
#    Reference implementation: all attention matrices
#    are materialized.


dV_naive = P.T @ dO

dP = dO @ V.T

dS = P * (
    dP - (dP * P).sum(dim=-1, keepdim=True)
)

dQ_naive = dS @ K / math.sqrt(d)
dK_naive = dS.T @ Q / math.sqrt(d)

# -----------------------------------------------------------
# 3. Reconstruct P block-by-block
#
#    Instead of storing P = softmax(S), reconstruct each
#    P_block using the m and l saved during the forward pass.

P_blocks = []

for kv_start in range(0, N, block_size):

    K_block = K[kv_start:kv_start + block_size]

    S_block = Q @ K_block.T / math.sqrt(d)

    m_block = m
    l_block = l

    P_block = torch.exp(
        S_block - m_block[:, None]
    ) / l_block[:, None]

    P_blocks.append(P_block)

P_reconstructed = torch.cat(P_blocks, dim=1)

print(
    "P reconstruction:",
    torch.allclose(P_reconstructed, P)
)

# -----------------------------------------------------------
# 4. Blockwise dV
#
#    dV = P.T @ dO
#
#    Each K/V block produces its own dV_block, so there is
#    no cross-block accumulation.

dV_blocks = []

for kv_start in range(0, N, block_size):

    K_block = K[kv_start:kv_start + block_size]
    V_block = V[kv_start:kv_start + block_size]

    S_block = Q @ K_block.T / math.sqrt(d)

    P_block = torch.exp(
        S_block - m[:, None]
    ) / l[:, None]

    dV_block = P_block.T @ dO

    dV_blocks.append(dV_block)

dV_blockwise = torch.cat(dV_blocks, dim=0)

print(
    "dV blockwise:",
    torch.allclose(dV_blockwise, dV_naive)
)

# -----------------------------------------------------------
# 5. Blockwise dQ and dK: first pass
#
#    dS requires:
#
#        D = rowsum(dP * P)
#
#    This reduction covers the ENTIRE attention row, so we
#    accumulate it across all K/V blocks.

D = torch.zeros(N, 1)

for q_start in range(0, N, block_size):

    Q_block = Q[q_start:q_start + block_size]
    dO_block = dO[q_start:q_start + block_size]

    D_block = torch.zeros(Q_block.size(0), 1)

    for kv_start in range(0, N, block_size):

        K_block = K[kv_start:kv_start + block_size]
        V_block = V[kv_start:kv_start + block_size]

        S_block = Q_block @ K_block.T / math.sqrt(d)

        m_block = m[q_start:q_start + block_size]
        l_block = l[q_start:q_start + block_size]

        P_block = torch.exp(
            S_block - m_block[:, None]
        ) / l_block[:, None]

        dP_block = dO_block @ V_block.T

        D_block += (
            dP_block * P_block
        ).sum(dim=-1, keepdim=True)

    D[q_start:q_start + block_size] = D_block

# -----------------------------------------------------------
# 6. Blockwise dQ, dK, dV: second pass
#
#    Now that D = rowsum(dP * P) is known, we can compute
#    dS_block and accumulate/write the three gradients.

dQ = torch.zeros_like(Q)
dK = torch.zeros_like(K)
dV = torch.zeros_like(V)

for q_start in range(0, N, block_size):

    Q_block = Q[q_start:q_start + block_size]
    dO_block = dO[q_start:q_start + block_size]

    m_block = m[q_start:q_start + block_size]
    l_block = l[q_start:q_start + block_size]
    D_block = D[q_start:q_start + block_size]

    for kv_start in range(0, N, block_size):

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

        # Every K/V block contributes to this Q block.
        dQ[q_start:q_start + block_size] += (
            dS_block @ K_block / math.sqrt(d)
        )

        # Every Q block contributes to this K block.
        dK[kv_start:kv_start + block_size] += (
            dS_block.T @ Q_block / math.sqrt(d)
        )

        # Every Q block contributes to this V block.
        dV[kv_start:kv_start + block_size] += (
            P_block.T @ dO_block
        )

# -----------------------------------------------------------
# 7. Validation
#
#    The blockwise implementation should match the naive
#    implementation up to floating-point error.

print("\nGradient validation:")

print("dQ:", torch.allclose(dQ, dQ_naive))
print("dK:", torch.allclose(dK, dK_naive))
print("dV:", torch.allclose(dV, dV_naive))

print("\nMaximum absolute differences:")

print("dQ:", (dQ - dQ_naive).abs().max())
print("dK:", (dK - dK_naive).abs().max())
print("dV:", (dV - dV_naive).abs().max())