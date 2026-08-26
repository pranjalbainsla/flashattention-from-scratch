import torch
from tiled_attention import tiled_attention

N = 10
d = 4

Q = torch.randn(N, d)
K = torch.randn(N, d)
V = torch.randn(N, d)

O_ref = torch.softmax(Q @ K.T, dim=-1) @ V

O_tiled = tiled_attention(
    Q, K, V,
    q_block_size=4,
    kv_block_size=3
)

print(torch.allclose(O_ref, O_tiled, atol=1e-6))
print(torch.max(torch.abs(O_ref - O_tiled)))