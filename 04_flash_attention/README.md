# FlashAttention forward pass

- `flash_attention.py` combines tiled attention and online softmax into a single reference implementation of the FlashAttention forward pass.  
- It adds the standard 1/√d score scaling and support for causal masking, where future-token scores are set to -inf.  
- The implementation also handles arbitrary Q and K/V block sizes, including partial blocks when the sequence length isn't divisible by the block size.  
> `flash_attention.py` focuses on algorithmic clarity and correctness rather than GPU performance