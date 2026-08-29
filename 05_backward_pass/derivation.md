
Forward:

```math
S = \frac{QK^\top}{\sqrt{d}}
```

```math
P = \text{softmax}(S)
```

```math
O = PV
```

Backward from `O = PV`:

```math
\boxed{
dV = P^\top dO
}
```

```math
\boxed{
dP = dO V^\top
}
```


## Softmax Backward Pass

For one row of scores:

```math
p_i = \frac{e^{s_i}}{\sum_k e^{s_k}}
```

Let:

```math
Z = \sum_k e^{s_k}
```

The key point is that changing one score $s_j$ affects **every** probability because every probability shares the same denominator.

### Derivative of softmax

There are two cases.

#### Case 1: $i = j$

How does $p_i$ change when we change its own score $s_i$?

```math
\frac{\partial p_i}{\partial s_i}
=
p_i(1-p_i)
```

The probability increases because its own numerator increases.

#### Case 2: $i \neq j$

How does $p_i$ change when we change another score $s_j$?

```math
\frac{\partial p_i}{\partial s_j}
=
-p_i p_j
```

The probability decreases because increasing $s_j$ increases the shared denominator.

Both cases can be written as:

```math
\boxed{
\frac{\partial p_i}{\partial s_j}
=
p_i(\delta_{ij}-p_j)
}
```

where $\delta_{ij}=1$ if $i=j$ and $0$ otherwise.

### Backpropagation

Suppose we already have:

```math
dP_j = \frac{\partial L}{\partial p_j}
```

We want:

```math
dS_i = \frac{\partial L}{\partial s_i}
```

By the chain rule:

```math
dS_i
=
\sum_j
dP_j
\frac{\partial p_j}{\partial s_i}
```

Substituting the softmax derivative:

```math
dS_i
=
\sum_j dP_j p_j(\delta_{ij}-p_i)
```

Splitting the two terms:

```math
dS_i
=
dP_i p_i
-
p_i\sum_j dP_jp_j
```

Therefore:

```math
\boxed{
dS_i
=
p_i
\left(
dP_i-\sum_jdP_jp_j
\right)
}
```

### Vectorized form

For an entire row:

```math
\boxed{
dS
=
P \odot
\left(
dP-\sum_j(dP_jP_j)
\right)
}
```

---
Backward from $S = \frac{QK^T}{\sqrt{d}}$:

Think of this as two operations:

```text
Q ──┐
    ├── QKᵀ ── /√d ──→ S
K ──┘
```

For ordinary matrix multiplication,

```math
S = QK^T
```

the gradients are:

```math
dQ = dS K
```

```math
dK = dS^T Q
```

The scaling by $\frac{1}{\sqrt{d}}$ simply carries through:

```math
\boxed{
dQ = \frac{dS K}{\sqrt{d}}
}
```

and

```math
\boxed{
dK = \frac{dS^T Q}{\sqrt{d}}
}
```

