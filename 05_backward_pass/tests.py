import torch
import math

from naive_backward import naive_backward
from flash_attention_backward import flash_attention_backward


torch.manual_seed(42)

N = 8
d = 4
block_size = 2

Q = torch.randn(N, d)
K = torch.randn(N, d)
V = torch.randn(N, d)

dO = torch.randn(N, d)


# ---------- Forward ----------

S = Q @ K.T / math.sqrt(d)
P = torch.softmax(S, dim=-1)

m = S.max(dim=-1).values
l = torch.exp(S - m[:, None]).sum(dim=-1)


# ---------- Naive reference ----------

dQ_naive, dK_naive, dV_naive = naive_backward(Q, K, V, P, dO)

# ---------- FlashAttention backward ----------

dQ, dK, dV = flash_attention_backward(Q, K, V, dO, m, l, block_size)

# ---------- Tests ----------

print("dQ:", torch.allclose(dQ, dQ_naive))
print("dK:", torch.allclose(dK, dK_naive))
print("dV:", torch.allclose(dV, dV_naive))

print(
    "max error dQ:",
    (dQ - dQ_naive).abs().max()
)

print(
    "max error dK:",
    (dK - dK_naive).abs().max()
)

print(
    "max error dV:",
    (dV - dV_naive).abs().max()
)
