#Vibe-coded speculative decoding benchmarking code for the `gpt2-xl` and `gpt2-large` models.
import csv
import json
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import tiktoken
import torch
from infer_speculative import generate, generate_speculative
from model import GPT

# -------------------------------------------------------------------------#
# Benchmark Configuration
# -------------------------------------------------------------------------#

BENCHMARK_CONFIGS = {
    # Model configurations to test
    "model_pairs": [
        {"target": "gpt2-xl", "draft": "gpt2"},
        {"target": "gpt2-large", "draft": "gpt2"},
        # Add more model pairs as needed
    ],
    
    # Gamma values to test
    "gamma_values": [3, 4, 5, 6, 7],
    
    # Data types to test
    "dtypes": ["float16", "float32", "bfloat16"],  # Options: "float16", "bfloat16", "float32"
    
    # Generation parameters
    "num_samples": 3,  # Number of runs per configuration for averaging
    "max_new_tokens": 200,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.9,
    # Note: generate functions always use sampling (no greedy decoding mode)
    
    # Test prompts
    "test_prompts": [
        "The following is a short story about a cat:",
        "Once upon a time in a distant galaxy,",
        "The future of artificial intelligence is",
    ],
    
    # Other settings
    "device": "cuda",
    "seed": 1337,
}

# -------------------------------------------------------------------------#
# Benchmark Functions
# -------------------------------------------------------------------------#

def benchmark_configuration(target_model, draft_model, enc, config, ctx, device):
    """Benchmark a single configuration."""
    
    # Handle None gamma value for baseline
    gamma_value = config.get("gamma")
    gamma_value = gamma_value if gamma_value is not None else "baseline"
    
    results = {
        "target_model": config["target_model_name"],
        "draft_model": config["draft_model_name"],
        "gamma": gamma_value,
        "dtype": config["dtype"],
        "prompt": config["prompt"],
        "runs": []
    }
    
    # Encode prompt
    tokens = enc.encode(config["prompt"], allowed_special={"<|endoftext|>"})
    x = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]
    
    for run_idx in range(config["num_samples"]):
        # Reset seed for each run to ensure consistency
        torch.manual_seed(config["seed"] + run_idx)
        if device == "cuda":
            torch.cuda.manual_seed(config["seed"] + run_idx)
        
        with ctx:
            start_time = time.time()
            
            if config.get("gamma") is None:
                # Baseline generation (using imported generate function)
                y = generate(
                    target_model, x, config["max_new_tokens"],
                    temperature=config["temperature"],
                    top_k=config["top_k"],
                    top_p=config["top_p"]
                )
                acceptance_ratio = None
            else:
                # Speculative generation
                y, acceptance_ratio = generate_speculative(
                    target_model, draft_model, x, config["max_new_tokens"],
                    gamma=config["gamma"],
                    temperature=config["temperature"],
                    top_k=config["top_k"],
                    top_p=config["top_p"]
                )
            
            # Synchronize for accurate timing
            if device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
        
        # Calculate metrics
        elapsed_time = end_time - start_time
        num_tokens_generated = y.size(1) - x.size(1)
        tokens_per_second = num_tokens_generated / elapsed_time if elapsed_time > 0 else 0
        
        run_result = {
            "run": run_idx + 1,
            "tokens_generated": num_tokens_generated,
            "time_seconds": elapsed_time,
            "tokens_per_second": tokens_per_second,
            "acceptance_ratio": acceptance_ratio,
            "generated_text": enc.decode(y[0,:].tolist())
        }
        
        results["runs"].append(run_result)
    
    # Calculate averages
    avg_tokens_per_second = sum(r["tokens_per_second"] for r in results["runs"]) / len(results["runs"])
    avg_time = sum(r["time_seconds"] for r in results["runs"]) / len(results["runs"])
    avg_acceptance = None
    if results["runs"][0]["acceptance_ratio"] is not None:
        avg_acceptance = sum(r["acceptance_ratio"] for r in results["runs"]) / len(results["runs"])
    
    results["averages"] = {
        "tokens_per_second": avg_tokens_per_second,
        "time_seconds": avg_time,
        "acceptance_ratio": avg_acceptance
    }
    
    return results

def run_benchmark():
    """Run the complete benchmark suite."""
    
    config = BENCHMARK_CONFIGS
    device = config["device"]
    
    # Setup device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("Warning: CUDA not available, using CPU")
    elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        device = "cpu"
        print("Warning: MPS not available, using CPU")
    
    print(f"\n{'='*100}")
    print(f"Starting Speculative Decoding Benchmark")
    print(f"Device: {device}")
    print(f"Testing dtypes: {', '.join(config['dtypes'])}")
    print(f"{'='*100}\n")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    all_results = []
    
    # Test each dtype
    for dtype in config["dtypes"]:
        # Check dtype support
        if device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            print(f"Warning: bfloat16 not supported on this device, skipping...")
            continue
        
        # Setup precision
        pdtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        ctx = nullcontext() if device == "cpu" or device == "mps" else torch.amp.autocast(device_type=device, dtype=pdtype)
        
        print(f"\n{'#'*100}")
        print(f"# Testing with dtype: {dtype}")
        print(f"{'#'*100}\n")
        
        # Test each model pair
        for model_pair in config["model_pairs"]:
            target_model_name = model_pair["target"]
            draft_model_name = model_pair["draft"]
            
            print(f"\n{'='*100}")
            print(f"Testing Model Pair: {target_model_name} (target) + {draft_model_name} (draft)")
            print(f"{'='*100}\n")
            
            # Load models
            print(f"Loading {target_model_name}...")
            target_model = GPT.from_pretrained(target_model_name)
            target_model.eval()
            target_model.to(device)
            
            print(f"Loading {draft_model_name}...")
            draft_model = GPT.from_pretrained(draft_model_name)
            draft_model.eval()
            draft_model.to(device)
            
            # Test baseline (no speculative decoding)
            print(f"\n{'-'*100}")
            print(f"Baseline (No Speculative Decoding)")
            print(f"{'-'*100}")
            
            for prompt in config["test_prompts"]:
                print(f"\nPrompt: '{prompt}'")
                
                test_config = {
                    "target_model_name": target_model_name,
                    "draft_model_name": draft_model_name,
                    "gamma": None,
                    "dtype": dtype,
                    "prompt": prompt,
                    "num_samples": config["num_samples"],
                    "max_new_tokens": config["max_new_tokens"],
                    "temperature": config["temperature"],
                    "top_k": config["top_k"],
                    "top_p": config["top_p"],
                    "seed": config["seed"]
                }
                
                result = benchmark_configuration(target_model, draft_model, enc, test_config, ctx, device)
                all_results.append(result)
                
                print(f"  Average: {result['averages']['tokens_per_second']:.2f} tokens/s")
            
            # Test different gamma values
            for gamma in config["gamma_values"]:
                print(f"\n{'-'*100}")
                print(f"Speculative Decoding (gamma={gamma})")
                print(f"{'-'*100}")
                
                for prompt in config["test_prompts"]:
                    print(f"\nPrompt: '{prompt}'")
                    
                    test_config = {
                        "target_model_name": target_model_name,
                        "draft_model_name": draft_model_name,
                        "gamma": gamma,
                        "dtype": dtype,
                        "prompt": prompt,
                        "num_samples": config["num_samples"],
                        "max_new_tokens": config["max_new_tokens"],
                        "temperature": config["temperature"],
                        "top_k": config["top_k"],
                        "top_p": config["top_p"],
                        "seed": config["seed"]
                    }
                    
                    result = benchmark_configuration(target_model, draft_model, enc, test_config, ctx, device)
                    all_results.append(result)
                    
                    print(f"  Average: {result['averages']['tokens_per_second']:.2f} tokens/s | "
                          f"Acceptance: {result['averages']['acceptance_ratio']:.2%}")
            
            # Clean up models to free memory
            del target_model
            del draft_model
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"benchmark_{timestamp}.json"
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
    
    # Group by dtype, model pair, and gamma
    # Structure: {dtype: {(target, draft): {gamma: [results]}}}
    grouped_by_dtype = {}
    for result in results:
        dtype = result["dtype"]
        target = result["target_model"]
        draft = result["draft_model"]
        gamma = result["gamma"]
        
        if dtype not in grouped_by_dtype:
            grouped_by_dtype[dtype] = {}
        if (target, draft) not in grouped_by_dtype[dtype]:
            grouped_by_dtype[dtype][(target, draft)] = {}
        if gamma not in grouped_by_dtype[dtype][(target, draft)]:
            grouped_by_dtype[dtype][(target, draft)][gamma] = []
        
        grouped_by_dtype[dtype][(target, draft)][gamma].append(result)
    
    # Print tables for each dtype and model pair
    for dtype in sorted(grouped_by_dtype.keys()):
        print(f"\n{'#'*100}")
        print(f"# Data Type: {dtype.upper()}")
        print(f"{'#'*100}")
        
        for (target, draft) in sorted(grouped_by_dtype[dtype].keys()):
            print(f"\n{'-'*100}")
            print(f"Model Pair: {target} (target) + {draft} (draft)")
            print(f"{'-'*100}")
            
            gamma_results = grouped_by_dtype[dtype][(target, draft)]
            
            # Calculate averages across all prompts for each gamma
            aggregated = {}
            for gamma, results_list in gamma_results.items():
                avg_tokens_per_sec = sum(r["averages"]["tokens_per_second"] for r in results_list) / len(results_list)
                avg_time = sum(r["averages"]["time_seconds"] for r in results_list) / len(results_list)
                
                if results_list[0]["averages"]["acceptance_ratio"] is not None:
                    avg_acceptance = sum(r["averages"]["acceptance_ratio"] for r in results_list) / len(results_list)
                else:
                    avg_acceptance = None
                
                aggregated[gamma] = {
                    "tokens_per_second": avg_tokens_per_sec,
                    "time_seconds": avg_time,
                    "acceptance_ratio": avg_acceptance
                }
            
            # Print table header
            print(f"\n{'Gamma':<15} {'Tokens/s':<18} {'Time (s)':<15} {'Acceptance':<15} {'Speedup':<15}")
            print("-" * 100)
            
            # Print baseline first
            if "baseline" in aggregated:
                baseline_data = aggregated["baseline"]
                baseline_speed = baseline_data["tokens_per_second"]
                print(f"{'Baseline':<15} {baseline_speed:<18.2f} {baseline_data['time_seconds']:<15.2f} {'N/A':<15} {'1.00x':<15}")
            else:
                baseline_speed = None
            
            # Print gamma values in order
            for gamma in sorted([g for g in aggregated.keys() if g != "baseline"]):
                data = aggregated[gamma]
                tokens_per_sec = data["tokens_per_second"]
                acceptance = data["acceptance_ratio"]
                time_sec = data["time_seconds"]
                
                if baseline_speed:
                    speedup = tokens_per_sec / baseline_speed
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "N/A"
                
                acceptance_str = f"{acceptance:.2%}" if acceptance is not None else "N/A"
                
                print(f"{str(gamma):<15} {tokens_per_sec:<18.2f} {time_sec:<15.2f} {acceptance_str:<15} {speedup_str:<15}")
            
            print("-" * 100)
    
    print("\n" + "="*100 + "\n")

def save_summary_table(results, results_dir, timestamp):
    """Save a single comprehensive summary CSV file with all benchmark results."""
    
    # Group by dtype, model pair, and gamma
    grouped_by_dtype = {}
    for result in results:
        dtype = result["dtype"]
        target = result["target_model"]
        draft = result["draft_model"]
        gamma = result["gamma"]
        
        if dtype not in grouped_by_dtype:
            grouped_by_dtype[dtype] = {}
        if (target, draft) not in grouped_by_dtype[dtype]:
            grouped_by_dtype[dtype][(target, draft)] = {}
        if gamma not in grouped_by_dtype[dtype][(target, draft)]:
            grouped_by_dtype[dtype][(target, draft)][gamma] = []
        
        grouped_by_dtype[dtype][(target, draft)][gamma].append(result)
    
    # Create a single CSV file with all results
    csv_filename = f"summary_all_{timestamp}.csv"
    csv_path = results_dir / csv_filename
    
    all_rows = []
    
    # Process each dtype and model pair
    for dtype in sorted(grouped_by_dtype.keys()):
        for (target, draft) in sorted(grouped_by_dtype[dtype].keys()):
            gamma_results = grouped_by_dtype[dtype][(target, draft)]
            
            # Calculate baseline speed for this model pair
            baseline_speed = None
            if "baseline" in gamma_results:
                results_list = gamma_results["baseline"]
                baseline_speed = sum(r["averages"]["tokens_per_second"] for r in results_list) / len(results_list)
                avg_time = sum(r["averages"]["time_seconds"] for r in results_list) / len(results_list)
                
                all_rows.append({
                    "dtype": dtype,
                    "target_model": target,
                    "draft_model": draft,
                    "gamma": "baseline",
                    "tokens_per_second": baseline_speed,
                    "time_seconds": avg_time,
                    "acceptance_ratio": None,
                    "speedup": 1.0
                })
            
            # Process gamma values
            for gamma in sorted([g for g in gamma_results.keys() if g != "baseline"]):
                results_list = gamma_results[gamma]
                avg_tokens_per_sec = sum(r["averages"]["tokens_per_second"] for r in results_list) / len(results_list)
                avg_time = sum(r["averages"]["time_seconds"] for r in results_list) / len(results_list)
                
                if results_list[0]["averages"]["acceptance_ratio"] is not None:
                    avg_acceptance = sum(r["averages"]["acceptance_ratio"] for r in results_list) / len(results_list)
                else:
                    avg_acceptance = None
                
                speedup = avg_tokens_per_sec / baseline_speed if baseline_speed else None
                
                all_rows.append({
                    "dtype": dtype,
                    "target_model": target,
                    "draft_model": draft,
                    "gamma": gamma,
                    "tokens_per_second": avg_tokens_per_sec,
                    "time_seconds": avg_time,
                    "acceptance_ratio": avg_acceptance,
                    "speedup": speedup
                })
    
    # Write single CSV with all results
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ["dtype", "target_model", "draft_model", "gamma", "tokens_per_second", 
                     "time_seconds", "acceptance_ratio", "speedup"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"Saved comprehensive summary table: {csv_path}")

# -------------------------------------------------------------------------#
# Main
# -------------------------------------------------------------------------#

if __name__ == "__main__":
    run_benchmark()

