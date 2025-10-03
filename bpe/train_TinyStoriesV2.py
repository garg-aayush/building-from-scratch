from bpe import BpeTokenizer

vocab_size = 276
tokenizer = BpeTokenizer()

# Read the training data
with open("data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
    text = f.read()

# Train the tokenizer on the actual text content
tokenizer.train(text, vocab_size, special_tokens=[], verbose=True)
tokenizer.register_special_tokens({"<|endoftext|>": vocab_size})
tokenizer.save(f"data/tinystoriesv2-gpt4-valid-baseline-{vocab_size}")
