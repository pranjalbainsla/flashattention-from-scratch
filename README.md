# quick note

This is my learning repo for FlashAttention. I'm building it up step by step, from plain attention to tiling, online softmax, and the blockwise backward pass, mostly to understand *why* it works. It's written for learning, so expect readable code and validation tests rather than a fast, production-ready kernel. It covers the FlashAttention 1 & 2 ideas (the memory IO story). FlashAttention 3 & 4 (warp specialization, async TMA/WGMMA, software pipelining) are hardware-level tricks for fancy GPUs that this repo doesn't try to do.

