# Tiling  

Suppose:

```text
N = 8
d = 4
```

Instead of computing all 8 queries against all 8 keys at once, divide them into blocks of 4.

Now, instead of producing:

```text
S = QKᵀ # 8 × 8
```

we produce four small score blocks:

```text
             K₀₋₃       K₄₋₇
          ┌─────────┬─────────┐
Q₀₋₃      │ 4 × 4   │ 4 × 4   │
          ├─────────┼─────────┤
Q₄₋₇      │ 4 × 4   │ 4 × 4   │
          └─────────┴─────────┘
```

Notice something important:

**We can compute one 4×4 block, use it, and throw it away.** We don't need to store all four blocks simultaneously.

This is the first major shift:

```text
Naive:

QKᵀ
 ↓
N × N
 ↓
store everything


Tiled:

Q block × K block
 ↓
small score block
 ↓
use it
 ↓
discard it
 ↓
next K block
```

But we've just created a new problem.

Suppose we're processing the first Q block:

```text
Q₀₋₃
```

We first process:

```text
Q₀₋₃ × K₀₋₃ᵀ
```

and then:

```text
Q₀₋₃ × K₄₋₇ᵀ
```

How do we combine the two blocks correctly?

We **cannot** simply do:

```text
softmax(block 1)
softmax(block 2)
```

and combine them.

Why?

Because softmax normalization happens across the **entire row**.

For example, suppose one query has scores:

```text
[4, 2, 7, 3]
```

and we split them:

```text
block 1 = [4, 2]
block 2 = [7, 3]
```

The correct softmax is:

```text
softmax([4, 2, 7, 3])
```

It is **not**:

```text
softmax([4, 2]) + softmax([7, 3])
```

because each block would normalize against a different denominator.

And this is exactly where the next major idea enters:

**online softmax.**

The question becomes:

> **Can we process one score block, throw it away, process the next score block, and still maintain exactly the same softmax normalization we would have gotten from seeing the entire row?**

(See [Online softmax theory](../03_online_softmax/README.md) before the next section)
# Tiled Attention

Standard attention materializes the full `N × N` score and probability matrices:

QKᵀ → softmax → P

The goal of tiled attention is to avoid materializing these matrices. Instead, we divide Q, K, and V into blocks and process them incrementally.

## Block structure

For a Q block $Q_i$ and K/V block $(K_j, V_j)$:

```math
S_{ij} = Q_i K_j^\top
```

The score block has shape: **(Bq, Bk)**

We process one K/V block at a time and immediately incorporate its contribution into the output.

Conceptually:

```text
Q block
    │
    ├── K/V block 0 → score block → update state
    ├── K/V block 1 → score block → update state
    ├── K/V block 2 → score block → update state
    └── ...
```

The score blocks are temporary. We never construct the complete `N × N` score matrix.

## Running state

For every query in the current Q block, we maintain:

m : running maximum score (Bq,)  
l : running softmax denominator (Bq,)  
A : running weighted sum of V (Bq, d)

When a new K/V block arrives:

```math
m_{\text{new}} = \max(m_{\text{old}},\ \text{block\_max})
```

```math
l_{\text{new}}
=
l_{\text{old}} \exp(m_{\text{old}} - m_{\text{new}})
+
\sum \exp(\text{score} - m_{\text{new}})
```

```math
A_{\text{new}}
=
A_{\text{old}} \exp(m_{\text{old}} - m_{\text{new}})
+
\sum \exp(\text{score} - m_{\text{new}})\,V
```
Finally:

```math
O_{\text{block}} = \frac{A}{l}
```

