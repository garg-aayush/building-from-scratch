# !/bin/bash
# python benchmark_encoding.py --tiktoken --special all > ../logslog_benchmark_encode_tiktoken_gpt2.txt
# python benchmark_encoding.py --tiktoken --tiktoken-model cl100k_base --special all > ../logslog_benchmark_encode_tiktoken_cl100k_base.txt 
python benchmark_encoding.py --special all > ../logslog_benchmark_encode_baseline.txt
# python benchmark_encoding.py --rust --special all > ../logslog_benchmark_encode_rust.txt