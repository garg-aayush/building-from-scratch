import argparse
import os

# Set environment variable for multiprocessing
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from utils.instruct_measures import (GSM8K, MMLU, SST, AlpacaEval,
                                     load_yaml_config, run_evaluation)

# Default config paths for all evaluators
ALL_CONFIGS = [
    "data/eval_configs/gsm8k_config.yaml",
    "data/eval_configs/mmlu_config.yaml",
    "data/eval_configs/sst_config.yaml",
    "data/eval_configs/alpaca_eval_config.yaml",
]

EVALUATOR_MAP = {
    "GSM8K": GSM8K,
    "MMLU": MMLU,
    "SST": SST,
    "AlpacaEval": AlpacaEval,
}


def run_single_evaluation(config_path: str):
    """Run evaluation for a single config file."""
    print(f"\n{'='*60}")
    print(f"Running evaluation with config: {config_path}")
    print(f"{'='*60}\n")
    
    # Load configuration
    config = load_yaml_config(config_path)
    
    # Extract parameters
    evaluator_name = config["evaluator"]
    evaluator_params = config["evaluator_params"]
    model_name = config["model_name"]
    sampling_params = config["sampling_params"]
    model_params = config["model_params"]
    model_params["model"] = model_name  # Add model name to model_params
    
    # Initialize evaluator based on type
    if evaluator_name not in EVALUATOR_MAP:
        raise ValueError(f"Unknown evaluator: {evaluator_name}")
    evaluator = EVALUATOR_MAP[evaluator_name](**evaluator_params)
    print(f"Initialized evaluator: {evaluator.__class__.__name__}")
    
    # Run evaluation using generic function
    results, metrics = run_evaluation(
        evaluator=evaluator,
        model_name=model_name,
        sampling_params=sampling_params,
        model_params=model_params,
    )
    
    return results, metrics


def main():
    """Run evaluation using configuration from YAML file."""
    parser = argparse.ArgumentParser(description="Run instruction evaluation")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--all", action="store_true", help="Run all 4 evaluations (GSM8K, MMLU, SST, AlpacaEval)")
    args = parser.parse_args()
    
    if args.all:
        # Run all evaluations
        all_results = {}
        for config_path in ALL_CONFIGS:
            try:
                results, metrics = run_single_evaluation(config_path)
            except Exception as e:
                print(f"Error running {config_path}: {e}")
    else:
        # Run single evaluation
        config_path = args.config or "data/eval_configs/gsm8k_config.yaml"
        run_single_evaluation(config_path)


if __name__ == "__main__":
    main()