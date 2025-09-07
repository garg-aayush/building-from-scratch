# Notes
## Multi-Head Attention
![Multi-Head Attention](../images/MHA.png)

- It's just applying multiple attentions in parallel and concatenating their results.

- It's fairly straightforward to implement it. If we want multi-head attention, then we want multiple heads of self-attention running in parallel. In PyTorch, we can do this by simply creating multiple heads - however many heads you want - and then determining the head size of each. We run all of them in parallel into a list and simply concatenate all of the outputs. We're concatenating over the channel dimension.

- The way this looks now is we don't have just a single attention head that has a head size of 32 (because remember `n_embed` is 32). Instead of having one communication channel, we now have four communication channels in parallel. Each one of these communication channels will typically be smaller correspondingly (n_embed//4, 8-dimensional vectors). 

- This is kind of similar to group convolutions if you're familiar with them - basically, instead of having one large convolution, we do convolution in groups. That's multi-headed self-attention.
- It helps to have multiple communication channels because obviously these tokens have a lot to talk about. They want to find the consonants, the vowels from certain positions, and any kinds of different things. So it helps to create multiple independent channels of communication, gather lots of different types of data, and then decode the output.