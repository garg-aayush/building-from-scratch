
from bpe import BpeTokenizer

if __name__ == '__main__':
    # vocab_size = 2048
    # vocab_size = 8192
    vocab_size = 16384
    # input_file_path = "data/TinyStoriesV2-GPT4-train.txt"
    input_file_path = "data/000_00003.txt"
    special_tokens = ["<|endoftext|>"]
    boundary_split_token = special_tokens[0]
    num_processes = 10 #os.cpu_count() 

    tokenizer = BpeTokenizer()

    # Train the tokenizer on the actual text content
    tokenizer.train(input_file_path=input_file_path, vocab_size=vocab_size, 
                    special_tokens=special_tokens, boundary_split_token=boundary_split_token, 
                    num_processes=num_processes, verbose=True
                    )
    tokenizer.register_special_tokens({"<|endoftext|>": vocab_size})
    tokenizer.save(f"data/fineweb-000_00003-gpt4-train-{vocab_size}")
    tokenizer.save(f"data/fineweb-000_00003-gpt4-train-{vocab_size}")
