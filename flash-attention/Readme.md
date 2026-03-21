# Flash Attention implementation from scratch

## Tasks

- [ ] Understand why vanilla attention is IO-bound, GPU memory hierarchy, tiling, online softmax, and the logsumexp trick
- [ ] Implement FlashAttention forward pass in pure PyTorch using `torch.autograd.Function`
- [ ] Write the Triton forward kernel with block pointers and causal masking
- [ ] Implement the backward pass — D vector trick, PyTorch + `torch.compile`, then Triton backward kernel
- [ ] (Stretch) Implement FA2 in NVIDIA cuTile on Blackwell
