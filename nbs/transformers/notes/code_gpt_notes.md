# Notes
## Multi-Head Attention
![Multi-Head Attention](../images/MHA.png)

- It's just applying multiple attentions in parallel and concatenating their results.

- It's fairly straightforward to implement it. If we want multi-head attention, then we want multiple heads of self-attention running in parallel. In PyTorch, we can do this by simply creating multiple heads - however many heads you want - and then determining the head size of each. We run all of them in parallel into a list and simply concatenate all of the outputs. We're concatenating over the channel dimension.

- The way this looks now is we don't have just a single attention head that has a head size of 32 (because remember `n_embed` is 32). Instead of having one communication channel, we now have four communication channels in parallel. Each one of these communication channels will typically be smaller correspondingly (n_embed//4, 8-dimensional vectors). 

- This is kind of similar to group convolutions if you're familiar with them - basically, instead of having one large convolution, we do convolution in groups. That's multi-headed self-attention.
- It helps to have multiple communication channels because obviously these tokens have a lot to talk about. They want to find the consonants, the vowels from certain positions, and any kinds of different things. So it helps to create multiple independent channels of communication, gather lots of different types of data, and then decode the output.

![Feed-Forward Networks](../images/FFN.png)

## Feed-Forward Networks

- Notice that there's a feed-forward part here, and then this is grouped into a block that gets repeated again and again. The feed-forward part here is just a simple multi-layer perceptron. The "position-wise feed-forward networks" is just a simple little MLP.

- I want to start adding computation into the network in a similar fashion, and this computation is on a per-node level. 

- Before, we had the multi-headed self-attention that did the communication, but we went way too fast to calculate the logits. The tokens looked at each other but didn't really have a lot of time to think about what they found from the other tokens. What I've implemented here is a little feed-forward single layer. This little layer is just a linear layer followed by a ReLU nonlinearity - that's it. It's just a little layer, and I call it `feed_forward` with `n_embed` dimensions.

- This feed-forward layer is called sequentially right after the self-attention. So we self-attention, then we feed-forward. You'll notice that the feed-forward here, when it's applying the linear transformation, operates on a per-token level - all the tokens do this independently. 

- So the self-attention is the communication phase, and then once they've gathered all the data, now they need to think on that data individually. That's what the feed-forward is doing.

## Residual Connections

- It is one of the two optimizations that dramatically help with the depth of these networks and make sure that the networks remain optimizable.

- The skip connections or sometimes called residual connections. They come from the "Deep Residual Learning for Image Recognition" paper from about 2015 that introduced the concept.

- These are basically what it means: you transform data but then you have a skip connection with addition from the previous features. The way I like to visualize it is the following: the computation happens from the top to bottom and basically you have this residual pathway. You are free to fork off from the residual pathway, perform some computation, and then project back to the residual pathway via addition. So you go from the inputs to the targets only via plus and plus and plus.

- The reason this is useful is because during backpropagation (remember from our micrograd video earlier), addition distributes gradients equally to both of its branches that fed as the input. So the supervision or the gradients from the loss basically hop through every addition node all the way to the input and then also fork off into the residual blocks. But basically you have this gradient superhighway that goes directly from the supervision all the way to the input unimpeded, and then these residual blocks are usually initialized in the beginning so they contribute very little if anything to the residual pathway.

- They are initialized that way so in the beginning they are sort of almost kind of like not there, but then during the optimization they come online over time and they start to contribute. But at least at the initialization you can go from directly supervision to the input - gradient is unimpeded and just flows, and then the blocks over time kick in. That dramatically helps with the optimization.

## Layer Normalization

- Layer normalization, is the second crutial optimizations and is a crucial component of the Transformer architecture.

- In batch normalization, we normalize across the batch dimension. For any individual neuron, we ensure it has zero mean and unit standard deviation across all examples in the batch. This means we're normalizing columns - each feature dimension is normalized independently across all samples.

- However, layer normalization works differently. Instead of normalizing across the batch dimension, we normalize across the feature dimension for each individual example. This means we're normalizing rows - each sample's features are normalized independently.

- The implementation is surprisingly simple. If we had batch normalization code that normalizes across dimension 0 (batch dimension), we just change it to normalize across dimension 1 (feature dimension) to get layer normalization.

- Layer normalization has several advantages over batch normalization in the context of Transformers:
    - **No dependency on batch size**: Layer norm works the same regardless of batch size, even with a batch size of 1
    - **Simpler implementation and tracking**: We don't need to maintain running mean and variance buffers. There's no distinction between training and test time
    - **Better for sequence models**: Each token's features are normalized independently, which is more appropriate for language modeling

## Post vs Pre Layer Normalization

- The original Transformer paper used **post-layer normalization**, where layer norm is applied after the multi-head attention and feed-forward operations.
- However, modern implementations have largely switched to **pre-layer normalization**, where layer norm is applied before the operations
- The switch to pre-layer normalization happened possibly due to:
    - **Better gradient flow**: Pre-layer norm creates a cleaner residual pathway, allowing gradients to flow more directly through the network
    - **More stable training**: The normalization happens before the potentially destabilizing operations (attention and feed-forward), leading to more stable gradients

## Scale and push the performance

One key addition is **Dropout**. Dropout is a regularization technique that can be added right before residual connections back into the residual pathway. Apply dropout:
- At the end of the multi-headed attention 
- When calculating the attention affinities after the softmax
- At various other points to randomly prevent some nodes from communicating

Dropout comes from a 2014 paper and works by randomly shutting off some subset of neurons during each forward/backward pass. The mask of what's being dropped out changes every iteration, effectively training an ensemble of sub-networks. At test time, everything is fully enabled and all those sub-networks merge into a single ensemble. This is a regularization technique I added because I was about to scale up the model significantly and was concerned about overfitting.

### Updated Hyperparameters

- **Batch size**: Increased to 64 (much larger than before)
- **Block size**: Increased to 256 characters of context (previously just 8 characters)
- **Learning rate**: Reduced slightly because the neural net is now much bigger
- **Embedding dimension**: Now 384 
- **Number of heads**: 6 heads (384 ÷ 6 = 64 dimensions per head, which is standard)
- **Number of layers**: 6 layers (Transformer Block)
- **Dropout**: 0.2 (20% of intermediate calculations are disabled each pass)

- After training this scaled-up model, the results were impressive. The validation loss improved to 1.48** - a significant improvement just from scaling up the neural network with our existing code. 
- The generated text is much more recognizable as Shakespeare-like output. While still nonsensical when you actually read, it maintains the characteristic structure and style of the input text - someone speaking in Shakespearean manner with proper formatting and dialogue structure. 
- This is just a character-level Transformer trained on 1 million characters from Shakespeare and a good demonstration of what's possible at this scale.