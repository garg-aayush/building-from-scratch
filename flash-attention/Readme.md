# FlashAttention Implementation from Scratch

I'm implementing FlashAttention 1 & 2 from first principles. I start from the conceptual foundations (why vanilla attention is IO-bound, how tiling and online softmax fix it) and build up one step at a time: PyTorch forward pass, Triton kernels, and the backward pass. As I go, I'm also writing a blog post for each step, both for my own reference and for anyone who wants to follow along.

## Tasks

- [ ] Understand why vanilla attention is IO-bound, GPU memory hierarchy, tiling, online softmax, and the logsumexp trick
- [ ] End-to-end profiling, mixed precision, memory analysis, and attention benchmarking
- [ ] Implement FlashAttention forward pass in pure PyTorch using `torch.autograd.Function`
- [ ] Learn Triton programming model,weighted sum kernel, online softmax as a fused GPU kernel with benchmarks
- [ ] Write the FlashAttention Triton forward kernel with block pointers and causal masking
- [ ] Implement the backward pass,D vector trick, recomputation-based backward in PyTorch + `torch.compile`
