import csv
import json
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import tiktoken
import torch
from infer import generate
from model import GPT

# -------------------------------------------------------------------------#
# Benchmark Configuration
# -------------------------------------------------------------------------#

BENCHMARK_CONFIGS = {
    # Model to test
    "model_name": "gpt2-xl",
    
    # Max new tokens to test
    "max_new_tokens_list": [200, 400, 800],
    
    # Data types to test (CUDA only)
    "dtypes": ["bfloat16", "float16", "float32"],
    
    # Generation parameters
    "num_samples": 3,  # Number of runs per configuration for averaging
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.9,
    "do_sample": True,
    
    # Test prompts
    "test_prompts": [
        "The following is a short story about a cat:",
        "Once upon a time in a distant galaxy,",
        "The future of artificial intelligence is",
    ],
    
    # Other settings
    "seed": 1337,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
}

# -------------------------------------------------------------------------#
# Helper Functions
# -------------------------------------------------------------------------#

def get_memory_usage(device):
    """Get current memory usage in MB."""
    if device == "cuda":
        return torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    else:
        # For CPU, we can't easily track memory per tensor, return 0
        return 0.0

def get_peak_memory_usage(device):
    """Get peak memory usage in MB."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
    else:
        return 0.0

def reset_peak_memory_stats(device):
    """Reset peak memory statistics."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

# -------------------------------------------------------------------------#
# Benchmark Functions
# -------------------------------------------------------------------------#

def benchmark_configuration(model, enc, config, ctx, device):
    """Benchmark a single configuration."""
    
    results = {
        "model_name": config["model_name"],
        "max_new_tokens": config["max_new_tokens"],
        "use_cache": config["use_cache"],
        "dtype": config["dtype"],
        "device": device,
        "prompt": config["prompt"],
        "runs": []
    }
    
    # Encode prompt
    tokens = enc.encode(config["prompt"], allowed_special={"<|endoftext|>"})
    x = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]
    
    # Prepare attention mask and initial lengths for the generate function
    initial_lengths = torch.tensor([x.size(1)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(x, dtype=torch.bool, device=device)
    
    for run_idx in range(config["num_samples"]):
        # Reset seed for each run to ensure consistency
        torch.manual_seed(config["seed"] + run_idx)
        if device == "cuda":
            torch.cuda.manual_seed(config["seed"] + run_idx)
        
        # Reset peak memory stats
        reset_peak_memory_stats(device)
        
        # Measure memory before generation
        mem_before = get_memory_usage(device)
        
        with ctx:
            start_time = time.time()
            
            y, final_lengths = generate(
                model, x, config["max_new_tokens"],
                attention_mask, initial_lengths,
                temperature=config["temperature"],
                do_sample=config["do_sample"],
                top_k=config["top_k"],
                top_p=config["top_p"],
                use_cache=config["use_cache"],
                presence_penalty=config["presence_penalty"],
                frequency_penalty=config["frequency_penalty"],
                repetition_penalty=config["repetition_penalty"]
            )
            
            # Synchronize for accurate timing
            if device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
        
        # Measure memory after generation
        mem_after = get_memory_usage(device)
        peak_mem = get_peak_memory_usage(device)
        
        # Calculate metrics
        elapsed_time = end_time - start_time
        num_tokens_generated = (final_lengths[0] - initial_lengths[0]).item()
        tokens_per_second = num_tokens_generated / elapsed_time if elapsed_time > 0 else 0
        
        run_result = {
            "run": run_idx + 1,
            "tokens_generated": num_tokens_generated,
            "time_seconds": elapsed_time,
            "tokens_per_second": tokens_per_second,
            "memory_before_mb": mem_before,
            "memory_after_mb": mem_after,
            "peak_memory_mb": peak_mem,
            "memory_delta_mb": mem_after - mem_before,
            "generated_text": enc.decode(y[0, :final_lengths[0]].tolist())
        }
        
        results["runs"].append(run_result)
        
        # Clear KV cache after each run
        model.clear_kv_cache()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Calculate averages
    avg_tokens_per_second = sum(r["tokens_per_second"] for r in results["runs"]) / len(results["runs"])
    avg_time = sum(r["time_seconds"] for r in results["runs"]) / len(results["runs"])
    avg_peak_memory = sum(r["peak_memory_mb"] for r in results["runs"]) / len(results["runs"])
    avg_memory_delta = sum(r["memory_delta_mb"] for r in results["runs"]) / len(results["runs"])
    
    results["averages"] = {
        "tokens_per_second": avg_tokens_per_second,
        "time_seconds": avg_time,
        "peak_memory_mb": avg_peak_memory,
        "memory_delta_mb": avg_memory_delta
    }
    
    return results

def run_benchmark():
    """Run the complete benchmark suite."""
    
    config = BENCHMARK_CONFIGS
    device = "cuda"
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a CUDA-enabled GPU.")
    
    print(f"\n{'='*100}")
    print(f"Starting KV-Cache Benchmark")
    print(f"Model: {config['model_name']}")
    print(f"Device: {device.upper()}")
    print(f"{'='*100}\n")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    all_results = []
    
    # Test each dtype
    for dtype in config["dtypes"]:
        # Check dtype support
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            print(f"Warning: bfloat16 not supported on this device, skipping...")
            continue
        
        # Setup precision
        pdtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        ctx = torch.amp.autocast(device_type=device, dtype=pdtype)
        
        print(f"\n{'='*100}")
        print(f"Testing with dtype: {dtype}")
        print(f"{'='*100}\n")
        
        # Load model
        print(f"Loading {config['model_name']}...")
        model = GPT.from_pretrained(config["model_name"])
        model.eval()
        model.to(device)
        print(f"Model loaded successfully\n")
        
        # Test each max_new_tokens value
        for max_new_tokens in config["max_new_tokens_list"]:
            print(f"\n{'-'*100}")
            print(f"Max New Tokens: {max_new_tokens}")
            print(f"{'-'*100}")
            
            # Test baseline (no KV cache)
            print(f"\nBaseline (No KV Cache)")
            print(f"{'.'*50}")
            
            for prompt in config["test_prompts"]:
                print(f"Prompt: '{prompt[:50]}...'")
                
                test_config = {
                    "model_name": config["model_name"],
                    "max_new_tokens": max_new_tokens,
                    "use_cache": False,
                    "dtype": dtype,
                    "prompt": prompt,
                    "num_samples": config["num_samples"],
                    "temperature": config["temperature"],
                    "do_sample": config["do_sample"],
                    "top_k": config["top_k"],
                    "top_p": config["top_p"],
                    "seed": config["seed"],
                    "presence_penalty": config["presence_penalty"],
                    "frequency_penalty": config["frequency_penalty"],
                    "repetition_penalty": config["repetition_penalty"]
                }
                
                result = benchmark_configuration(model, enc, test_config, ctx, device)
                all_results.append(result)
                
                print(f"  Average: {result['averages']['tokens_per_second']:.2f} tokens/s, "
                      f"Peak Memory: {result['averages']['peak_memory_mb']:.2f} MB")
            
            # Test with KV cache
            print(f"\nWith KV Cache")
            print(f"{'.'*50}")
            
            for prompt in config["test_prompts"]:
                print(f"Prompt: '{prompt[:50]}...'")
                
                test_config = {
                    "model_name": config["model_name"],
                    "max_new_tokens": max_new_tokens,
                    "use_cache": True,
                    "dtype": dtype,
                    "prompt": prompt,
                    "num_samples": config["num_samples"],
                    "temperature": config["temperature"],
                    "do_sample": config["do_sample"],
                    "top_k": config["top_k"],
                    "top_p": config["top_p"],
                    "seed": config["seed"],
                    "presence_penalty": config["presence_penalty"],
                    "frequency_penalty": config["frequency_penalty"],
                    "repetition_penalty": config["repetition_penalty"]
                }
                
                result = benchmark_configuration(model, enc, test_config, ctx, device)
                all_results.append(result)
                
                print(f"  Average: {result['averages']['tokens_per_second']:.2f} tokens/s, "
                      f"Peak Memory: {result['averages']['peak_memory_mb']:.2f} MB")
        
        # Clean up model to free memory
        del model
        torch.cuda.empty_cache()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"benchmark_kv_cache_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "config": config,
            "results": all_results,
            "timestamp": timestamp
        }, f, indent=2)
    
    print(f"\n{'='*100}")
    print(f"Benchmark Complete!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*100}\n")
    
    # Print summary
    print_summary(all_results)
    
    # Save summary table as CSV
    print(f"\n{'='*100}")
    print(f"Saving Summary Table as CSV...")
    print(f"{'='*100}\n")
    save_summary_table(all_results, results_dir, timestamp)
    
    return all_results

def print_summary(results):
    """Print a summary table of benchmark results."""
    
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY - AGGREGATED METRICS (Averaged Across All Prompts)")
    print("="*100)
    
    # Group by dtype, max_new_tokens, and use_cache
    # Structure: {dtype: {max_new_tokens: {use_cache: [results]}}}
    grouped = {}
    for result in results:
        dtype = result["dtype"]
        max_tokens = result["max_new_tokens"]
        use_cache = result["use_cache"]
        
        if dtype not in grouped:
            grouped[dtype] = {}
        if max_tokens not in grouped[dtype]:
            grouped[dtype][max_tokens] = {}
        if use_cache not in grouped[dtype][max_tokens]:
            grouped[dtype][max_tokens][use_cache] = []
        
        grouped[dtype][max_tokens][use_cache].append(result)
    
    # Print tables for each dtype
    for dtype in sorted(grouped.keys()):
        print(f"\n{'#'*100}")
        print(f"# Data Type: {dtype.upper()}")
        print(f"{'#'*100}")
        
        for max_tokens in sorted(grouped[dtype].keys()):
            print(f"\nMax New Tokens: {max_tokens}")
            print(f"{'.'*100}")
            
            cache_results = grouped[dtype][max_tokens]
            
            # Calculate averages across all prompts
            aggregated = {}
            for use_cache, results_list in cache_results.items():
                avg_tokens_per_sec = sum(r["averages"]["tokens_per_second"] for r in results_list) / len(results_list)
                avg_time = sum(r["averages"]["time_seconds"] for r in results_list) / len(results_list)
                avg_peak_mem = sum(r["averages"]["peak_memory_mb"] for r in results_list) / len(results_list)
                avg_mem_delta = sum(r["averages"]["memory_delta_mb"] for r in results_list) / len(results_list)
                
                aggregated[use_cache] = {
                    "tokens_per_second": avg_tokens_per_sec,
                    "time_seconds": avg_time,
                    "peak_memory_mb": avg_peak_mem,
                    "memory_delta_mb": avg_mem_delta
                }
            
            # Print table header
            print(f"\n{'Mode':<20} {'Tokens/s':<15} {'Time (s)':<15} {'Peak Mem (MB)':<20} {'Speedup':<15}")
            print("-" * 100)
            
            # Print baseline (no cache) first
            if False in aggregated:
                baseline_data = aggregated[False]
                baseline_speed = baseline_data["tokens_per_second"]
                print(f"{'No KV Cache':<20} {baseline_speed:<15.2f} {baseline_data['time_seconds']:<15.2f} "
                      f"{baseline_data['peak_memory_mb']:<20.2f} {'1.00x':<15}")
            else:
                baseline_speed = None
            
            # Print with cache
            if True in aggregated:
                cache_data = aggregated[True]
                cache_speed = cache_data["tokens_per_second"]
                
                if baseline_speed:
                    speedup = cache_speed / baseline_speed
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "N/A"
                
                print(f"{'With KV Cache':<20} {cache_speed:<15.2f} {cache_data['time_seconds']:<15.2f} "
                      f"{cache_data['peak_memory_mb']:<20.2f} {speedup_str:<15}")
            
            print("-" * 100)
    
    print("\n" + "="*100 + "\n")

def save_summary_table(results, results_dir, timestamp):
    """Save a comprehensive summary CSV file with all benchmark results."""
    
    # Group by dtype, max_new_tokens, and use_cache
    grouped = {}
    for result in results:
        dtype = result["dtype"]
        max_tokens = result["max_new_tokens"]
        use_cache = result["use_cache"]
        
        if dtype not in grouped:
            grouped[dtype] = {}
        if max_tokens not in grouped[dtype]:
            grouped[dtype][max_tokens] = {}
        if use_cache not in grouped[dtype][max_tokens]:
            grouped[dtype][max_tokens][use_cache] = []
        
        grouped[dtype][max_tokens][use_cache].append(result)
    
    # Create CSV file with all results
    csv_filename = f"summary_kv_cache_{timestamp}.csv"
    csv_path = results_dir / csv_filename
    
    all_rows = []
    
    # Process each dtype and max_tokens
    for dtype in sorted(grouped.keys()):
        for max_tokens in sorted(grouped[dtype].keys()):
            cache_results = grouped[dtype][max_tokens]
            
            # Calculate baseline speed
            baseline_speed = None
            if False in cache_results:
                results_list = cache_results[False]
                baseline_speed = sum(r["averages"]["tokens_per_second"] for r in results_list) / len(results_list)
                avg_time = sum(r["averages"]["time_seconds"] for r in results_list) / len(results_list)
                avg_peak_mem = sum(r["averages"]["peak_memory_mb"] for r in results_list) / len(results_list)
                avg_mem_delta = sum(r["averages"]["memory_delta_mb"] for r in results_list) / len(results_list)
                
                all_rows.append({
                    "dtype": dtype,
                    "max_new_tokens": max_tokens,
                    "use_cache": "No",
                    "tokens_per_second": baseline_speed,
                    "time_seconds": avg_time,
                    "peak_memory_mb": avg_peak_mem,
                    "memory_delta_mb": avg_mem_delta,
                    "speedup": 1.0
                })
            
            # Process with cache
            if True in cache_results:
                results_list = cache_results[True]
                avg_tokens_per_sec = sum(r["averages"]["tokens_per_second"] for r in results_list) / len(results_list)
                avg_time = sum(r["averages"]["time_seconds"] for r in results_list) / len(results_list)
                avg_peak_mem = sum(r["averages"]["peak_memory_mb"] for r in results_list) / len(results_list)
                avg_mem_delta = sum(r["averages"]["memory_delta_mb"] for r in results_list) / len(results_list)
                
                speedup = avg_tokens_per_sec / baseline_speed if baseline_speed else None
                
                all_rows.append({
                    "dtype": dtype,
                    "max_new_tokens": max_tokens,
                    "use_cache": "Yes",
                    "tokens_per_second": avg_tokens_per_sec,
                    "time_seconds": avg_time,
                    "peak_memory_mb": avg_peak_mem,
                    "memory_delta_mb": avg_mem_delta,
                    "speedup": speedup
                })
    
    # Write CSV with all results
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ["dtype", "max_new_tokens", "use_cache", "tokens_per_second", 
                     "time_seconds", "peak_memory_mb", "memory_delta_mb", "speedup"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"Saved comprehensive summary table: {csv_path}")

# -------------------------------------------------------------------------#
# Main
# -------------------------------------------------------------------------#

if __name__ == "__main__":
    run_benchmark()

