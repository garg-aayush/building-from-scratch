import os

from bpe import BpeTokenizer

if __name__ == '__main__':
    vocab_size = 276
    input_file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    boundary_split_token = special_tokens[0]
    num_processes = os.cpu_count() // 2

    tokenizer = BpeTokenizer()

    # Train the tokenizer on the actual text content
    tokenizer.train(input_file_path=input_file_path, vocab_size=vocab_size, 
                    special_tokens=special_tokens, boundary_split_token=boundary_split_token, 
                    num_processes=num_processes, verbose=True
                    )
    tokenizer.register_special_tokens({"<|endoftext|>": vocab_size})
    tokenizer.save(f"data/tinystoriesv2-gpt4-valid-optimized-pre-token-ll-{vocab_size}")
    tokenizer.save(f"data/tinystoriesv2-gpt4-valid-optimized-pre-token-ll-{vocab_size}")
