
# FlashAttention Backward Pass

We already know the forward pass:

```math
S = \frac{QK^T}{\sqrt{d}}
```

```math
P = softmax(S)
```

```math
O = PV
```

Now we want the gradients flowing backward from `O`:

```text
dO
 ↓
dV, dP
 ↓
dS
 ↓
dQ, dK
```

The standard backward equations are:

```math
dV = P^T dO
```

```math
dP = dO V^T
```

For softmax:

```math
dS = P \odot (dP - rowsum(dP \odot P))
```

And finally:

```math
dQ = \frac{dS K}{\sqrt{d}}
```

```math
dK = \frac{dS^T Q}{\sqrt{d}}
```

The naive implementation is straightforward, but it has the same problem as the naive forward pass: `P` is an `N × N` matrix.

So the goal is:

> Calculate the same gradients without ever materializing the full attention matrix.

## Reconstructing `P`

During the forward pass, we already saved the row-wise softmax statistics:

```text
m = row maximum
l = sum(exp(score - m))
```

Therefore, when we encounter a score block:

```text
S_block
```

we can reconstruct its corresponding probabilities:

```math
P_{block} =
\frac{e^{S_{block}-m}}{l}
```

The important point is that `m` and `l` are the **final statistics for the entire row**, calculated during the forward pass.

So backward doesn't need to maintain running `m` and `l` again.

It can simply:

```text
load Q block
      ↓
load K/V block
      ↓
calculate S_block
      ↓
reconstruct P_block
      ↓
use P_block
      ↓
discard P_block
```

## Blockwise `dV`

The naive equation is:

```math
dV = P^T dO
```

If we split `P` into K/V blocks:

```text
P = [P₁ P₂ P₃ ...]
```

then each block independently gives:

```math
dV_i = P_i^T dO
```

So we can calculate `dV` one block at a time.

No cross-block accumulation is needed for `dV` because each `P_block` corresponds to a distinct block of `V`.

## The Softmax Backward Problem

The interesting part is:

```math
dS = P \odot (dP - rowsum(dP \odot P))
```

The `rowsum` is over the **entire attention row**.

But we're processing K blocks separately.

Therefore, we first calculate:

```math
D_i = \sum_j dP_{ij}P_{ij}
```

by accumulating its contribution from every K block:

```text
K block 1 → contribution ─┐
K block 2 → contribution ─┤
K block 3 → contribution ─┤→ D
K block 4 → contribution ─┘
```

This means the backward pass needs a first pass over the blocks.

## Two-Pass Backward

Once `D` is known, we can make a second pass.

For each Q/K block:

```math
dS_{block}
=
P_{block}
\odot
(dP_{block} - D)
```

Then use it to update the gradients:

```math
dQ_{block}
\mathrel{+}=
\frac{dS_{block}K_{block}}{\sqrt{d}}
```

```math
dK_{block}
\mathrel{+}=
\frac{dS_{block}^TQ_{block}}{\sqrt{d}}
```

```math
dV_{block}
\mathrel{+}=
P_{block}^TdO_{block}
```

Notice the accumulation:

```text
dQ:
every K block contributes → accumulate

dK:
every Q block contributes → accumulate

dV:
every Q block contributes → accumulate
```

## The Final Picture

Instead of creating:

```text
S  → [N × N]
P  → [N × N]
dP → [N × N]
dS → [N × N]
```

we repeatedly work with small tiles:

```text
             K blocks
          ┌────┬────┬────┐
Q block 1 │    │    │    │
          ├────┼────┼────┤
Q block 2 │    │    │    │
          ├────┼────┼────┤
Q block 3 │    │    │    │
          ├────┼────┼────┤
Q block 4 │    │    │    │
          └────┴────┴────┘
```

For each tile:

```text
Q_block, K_block, V_block
          ↓
       S_block
          ↓
   reconstruct P_block
          ↓
        dP_block
          ↓
        dS_block
          ↓
      dQ, dK, dV
```

The full `N × N` attention matrix never needs to exist.

The important connection to the forward pass is:

> **Forward saves `m` and `l` so backward can reconstruct `P` when needed.**

That is what allows the backward pass to remain memory-efficient while producing the same gradients as the naive implementation.

The implementation was first built and validated as a naive reference against PyTorch autograd, then rewritten blockwise and validated against the naive implementation.


