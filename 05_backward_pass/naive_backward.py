import torch
import math

torch.manual_seed(42)

N = 8
d = 4

Q = torch.randn(N, d, requires_grad=True)
K = torch.randn(N, d, requires_grad=True)
V = torch.randn(N, d, requires_grad=True)

dO = torch.randn(N, d)

# ---------- Forward ----------

S = Q @ K.T / math.sqrt(d)
P = torch.softmax(S, dim=-1)
O = P @ V

# ------- Manual backward -------

def naive_backward(Q, K, V, P, dO):

    dV = P.T @ dO

    dP = dO @ V.T

    row_sum = (dP * P).sum(dim=-1, keepdim=True)

    dS = P * (dP - row_sum)

    dQ = dS @ K / math.sqrt(d)
    dK = dS.T @ Q / math.sqrt(d)

    return dQ, dK, dV

# ---------- Autograd verification ----------

if __name__ == "__main__":

    dQ_manual, dK_manual, dV_manual = naive_backward(
        Q, K, V, P, dO
    )

    L = (O * dO).sum()

    L.backward()

    dQ_autograd = Q.grad
    dK_autograd = K.grad
    dV_autograd = V.grad

    print("Naive backward vs autograd:")

    print("dQ:", torch.allclose(dQ_manual, dQ_autograd))
    print("dK:", torch.allclose(dK_manual, dK_autograd))
    print("dV:", torch.allclose(dV_manual, dV_autograd))

    print(
        "max error dQ:",
        (dQ_manual - dQ_autograd).abs().max()
    )

    print(
        "max error dK:",
        (dK_manual - dK_autograd).abs().max()
    )

    print(
        "max error dV:",
        (dV_manual - dV_autograd).abs().max()
    )