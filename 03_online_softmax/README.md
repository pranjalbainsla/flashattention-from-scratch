# Online Softmax

The problem is:

```text
scores = [4, 2, 7, 3]
```

We want:

```math
\text{softmax}(x_i)
=
\frac{e^{x_i}}{\sum_j e^{x_j}}
```

But for numerical stability, we actually calculate:

```math
\text{softmax}(x_i)
=
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
```

where:

```math
m = \max_j x_j
```

For the whole row:

```text
x = [4, 2, 7, 3]

m = 7
```

So we'd calculate:

```text
exp(4 - 7)
exp(2 - 7)
exp(7 - 7)
exp(3 - 7)
```

The problem is that with tiling, we don't see all the scores at once.

Suppose we see:

```text
block 1 = [4, 2]
```

We can calculate:

```math
m_1 = 4
```

and:

```math
l_1 = e^{4-4} + e^{2-4}
```

Then we encounter:

```text
block 2 = [7, 3]
```

Now the maximum changed:

```math
m_2 = 7
```

So here's the problem:

We previously calculated everything relative to `m₁ = 4`, but now we need everything relative to `m₂ = 7`.

**We can fix this mathematically.**

The old normalization was:

```math
l_1
=
\sum_{j \in block1} e^{x_j-m_1}
```

The new normalization should be:

```math
l_2
=
\sum_{j \in block1 \cup block2} e^{x_j-m_2}
```

For the old block:

```math
e^{x_j-m_2}
=
e^{x_j-m_1}e^{m_1-m_2}
```

Therefore:

```math
l_2
=
l_1 e^{m_1-m_2}
+
\sum_{j \in block2} e^{x_j-m_2}
```

That's the crucial recurrence.

We maintain:

```text
m = running maximum
l = running sum of exponentials
```

When a new block arrives:

```text
m_new = max(m_old, block_max)

l_new =
    l_old * exp(m_old - m_new)
    +
    block_sum
```

For our example:

```text
block 1 = [4, 2]

m_old = 4
l_old = exp(0) + exp(-2)
      ≈ 1.1353
```

Then:

```text
block 2 = [7, 3]

block_max = 7

m_new = max(4, 7)
      = 7
```

So the old contribution needs to be rescaled:

```text
l_old * exp(4 - 7)
```

and then we add the contribution from `[7, 3]`.

The result is exactly the same normalization we'd get if we'd processed `[4, 2, 7, 3]` all at once.

This is the fundamental trick that allows FlashAttention to process attention in tiles.

But there's one more piece.

Remember earlier we established that we need:

```math
O_i
=
\frac{
\sum_j e^{S_{ij}-m_i}V_j
}{
\sum_j e^{S_{ij}-m_i}
}
```

So maintaining `m` and `l` isn't enough.

We also need to maintain the numerator:

```math
A
=
\sum_j e^{S_{ij}-m}V_j
```

When a new block arrives, we'll update `A` using the exact same rescaling idea.

```math
A_{\text{new}}
=
A_{\text{old}} e^{m_{\text{old}}-m_{\text{new}}}
+
\sum_j e^{s_j-m_{\text{new}}}V_j
```

And finally:

```math
O = \frac{A}{l}
```

This is the core mathematical mechanism behind FlashAttention's forward pass.