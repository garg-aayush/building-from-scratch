# Notes
## Multi-Head Attention
![Multi-Head Attention](../images/MHA.png)

- It's just applying multiple attentions in parallel and concatenating their results.

- It's fairly straightforward to implement it. If we want multi-head attention, then we want multiple heads of self-attention running in parallel. In PyTorch, we can do this by simply creating multiple heads - however many heads you want - and then determining the head size of each. We run all of them in parallel into a list and simply concatenate all of the outputs. We're concatenating over the channel dimension.

- The way this looks now is we don't have just a single attention head that has a head size of 32 (because remember `n_embed` is 32). Instead of having one communication channel, we now have four communication channels in parallel. Each one of these communication channels will typically be smaller correspondingly (n_embed//4, 8-dimensional vectors). 

- This is kind of similar to group convolutions if you're familiar with them - basically, instead of having one large convolution, we do convolution in groups. That's multi-headed self-attention.
- It helps to have multiple communication channels because obviously these tokens have a lot to talk about. They want to find the consonants, the vowels from certain positions, and any kinds of different things. So it helps to create multiple independent channels of communication, gather lots of different types of data, and then decode the output.

## Feed-Forward Networks
![Feed-Forward Networks](../images/FFN.png)

Notice that there's a feed-forward part here, and then this is grouped into a block that gets repeated again and again. The feed-forward part here is just a simple multi-layer perceptron. The "position-wise feed-forward networks" is just a simple little MLP.

I want to start adding computation into the network in a similar fashion, and this computation is on a per-node level. 

Before, we had the multi-headed self-attention that did the communication, but we went way too fast to calculate the logits. The tokens looked at each other but didn't really have a lot of time to think about what they found from the other tokens. What I've implemented here is a little feed-forward single layer. This little layer is just a linear layer followed by a ReLU nonlinearity - that's it. It's just a little layer, and I call it `feed_forward` with `n_embed` dimensions.

This feed-forward layer is called sequentially right after the self-attention. So we self-attention, then we feed-forward. You'll notice that the feed-forward here, when it's applying the linear transformation, operates on a per-token level - all the tokens do this independently. 

So the self-attention is the communication phase, and then once they've gathered all the data, now they need to think on that data individually. That's what the feed-forward is doing.