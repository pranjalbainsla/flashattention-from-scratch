import torch

from flash_attention import flash_attention


def reference_attention(Q, K, V, causal=False):
    d = Q.size(-1)

    scores = Q @ K.T / (d ** 0.5)

    if causal:
        N = Q.size(0)
        positions = torch.arange(N, device=Q.device)
        mask = positions[:, None] < positions[None, :]
        scores = scores.masked_fill(mask, float("-inf"))

    P = torch.softmax(scores, dim=-1)

    return P @ V


def test_flash_attention(causal):
    torch.manual_seed(42)

    N = 10
    d = 16

    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)

    O_reference = reference_attention(Q, K, V, causal=causal)

    O_flash = flash_attention(
        Q,
        K,
        V,
        q_block_size=4,
        kv_block_size=3,
        causal=causal,
    )

    assert torch.allclose(
        O_flash,
        O_reference,
        atol=1e-5,
        rtol=1e-5,
    )


if __name__ == "__main__":
    test_flash_attention(causal=False)
    test_flash_attention(causal=True)

    print("All tests passed!")