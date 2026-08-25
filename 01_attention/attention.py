import torch

d = 16 # head dimension
# N = sequence length

for N in [4, 128, 512, 1024, 4096, 8192]:
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)

    # scores matrix
    S = torch.matmul(Q, K.T) / (d ** 0.5) # (N, N)

    # attention weights
    P = torch.softmax(S, dim=1) # (N, N)

    # output matrix
    O = torch.matmul(P, V) # (N, d)

    print(f"N={N}")
    print(f"Scores shape: {S.shape}")
    print(f"Elements: {S.numel():,}")
    print(f"Memory (fp32): {S.numel() * 4 / 1024**2:.2f} MB\n")
    # numbers same for P


