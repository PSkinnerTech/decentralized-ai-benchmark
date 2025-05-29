#!/usr/bin/env python3
import os
import sys
import argparse
import openai
from dotenv import load_dotenv
import time

# Import our metrics function
from hyperapi import get_model_metrics, MODEL_PRICING, get_api_client, PROVIDER_MODELS

# Import MMLU evaluation functions
try:
    from mmlu_eval import evaluate_model_on_mmlu, compare_mmlu_results, prepare_mmlu_test_cases
    from mmlu_dataset import DEFAULT_SUBJECTS, prepare_few_shot_examples, load_dataset
    MMLU_AVAILABLE = True
except ImportError:
    print("Warning: MMLU evaluation module not available")
    MMLU_AVAILABLE = False

# Load environment variables
load_dotenv()

# Default test prompts if no file is provided
DEFAULT_TEST_PROMPTS = [
    {"prompt": "Who wrote Hamlet?", "expected_keywords": ["Shakespeare"]},
    {"prompt": "What is 2 + 2?", "expected_keywords": ["4", "four"]},
    {"prompt": "What is the capital of France?", "expected_keywords": ["Paris"]},
]

def parse_prompts_file(file_path):
    """Parse a prompts file. Each line should be a prompt, optionally with expected answers after a | character."""
    prompts = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
                
            if '|' in line:
                # Format: "prompt | expected1, expected2"
                parts = line.split('|', 1)
                prompt = parts[0].strip()
                expected = [kw.strip() for kw in parts[1].split(',')]
                prompts.append({"prompt": prompt, "expected_keywords": expected})
            else:
                # Just a prompt with no expected answer
                prompts.append({"prompt": line.strip(), "expected_keywords": []})
    
    return prompts

def run_accuracy_tests(client, model_name, test_prompts, system_prompt, temperature=0, verbose=False, provider=None):
    """Run accuracy tests on a model and return metrics."""
    print(f"\n--- Testing Model: {model_name} ({provider or 'auto-detected provider'}) ---")
    correct_responses = 0
    total_prompts = len(test_prompts)
    total_cost = 0
    all_metrics = []
    
    for i, test_case in enumerate(test_prompts):
        print(f"\nRunning Test {i+1}/{total_prompts}: '{test_case['prompt']}'")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_case["prompt"]},
        ]
        
        metrics = get_model_metrics(client, model_name, messages, verbose=verbose, temperature=temperature, provider=provider)
        all_metrics.append(metrics)
        total_cost += metrics.get("cost", 0)

        is_correct = False
        if metrics.get("response_content"):
            # Print detailed metrics for this test case
            print(f"  TTFT: {metrics['ttft']:.4f}s" if metrics['ttft'] is not None else "  TTFT: N/A")
            print(f"  Latency: {metrics['latency']:.4f}s")
            print(f"  Prompt Tokens: {metrics['prompt_tokens']}")
            print(f"  Completion Tokens: {metrics['completion_tokens']}")
            print(f"  Throughput: {metrics['throughput']:.2f} tokens/sec")
            print(f"  Cost: ${metrics['cost']:.6f}")
            print(f"  Provider: {metrics.get('provider', 'unknown')}")
            
            if test_case["expected_keywords"]:
                response_lower = metrics["response_content"].lower()
                print(f"  Response Snippet: {metrics['response_content'][:80]}...") 
                for keyword in test_case["expected_keywords"]:
                    if keyword.lower() in response_lower:
                        is_correct = True
                        print(f"  Result: Correct (found '{keyword}')")
                        correct_responses += 1
                        break # Stop checking keywords for this prompt once one match is found
                if not is_correct:
                    print(f"  Result: Incorrect (expected keywords not found: {test_case['expected_keywords']})")
            else:
                # No expected keywords specified, just show the response
                print(f"  Response: {metrics['response_content'][:150]}...")
        else:
            print(f"  Result: Error - No response content found. Error: {metrics.get('error')}")

    # Calculate accuracy only if we have expected keywords
    has_expected_answers = any(test_case.get("expected_keywords") for test_case in test_prompts)
    accuracy_score = correct_responses / total_prompts if total_prompts > 0 and has_expected_answers else None
    
    # Calculate averages
    valid_metrics = [m for m in all_metrics if m.get("ttft") is not None]
    avg_ttft = sum(m["ttft"] for m in valid_metrics) / len(valid_metrics) if valid_metrics else None
    
    # Make sure to filter out None values for latency calculation
    valid_latency_metrics = [m for m in all_metrics if m.get("latency") is not None]
    avg_latency = sum(m["latency"] for m in valid_latency_metrics) / len(valid_latency_metrics) if valid_latency_metrics else None
    
    # Make sure to filter out None values for throughput calculation
    valid_throughput_metrics = [m for m in all_metrics if m.get("throughput") is not None]
    avg_throughput = sum(m["throughput"] for m in valid_throughput_metrics) / len(valid_throughput_metrics) if valid_throughput_metrics else 0
    
    # Summary
    summary = {
        "model": model_name,
        "provider": provider or all_metrics[0].get("provider", "unknown") if all_metrics else "unknown",
        "correct_responses": correct_responses,
        "total_prompts": total_prompts,
        "accuracy_score": accuracy_score,
        "total_cost": total_cost,
        "avg_ttft": avg_ttft,
        "avg_latency": avg_latency,
        "avg_throughput": avg_throughput,
        "pricing": MODEL_PRICING.get(model_name, MODEL_PRICING["default"])
    }
    
    # Print summary
    print("\n--- Model Summary ---")
    print(f"Model: {model_name}")
    print(f"Provider: {summary['provider']}")
    if has_expected_answers:
        print(f"Correct Responses: {correct_responses}/{total_prompts}")
        print(f"Accuracy Score: {accuracy_score:.2f}")
    print(f"Average TTFT: {avg_ttft:.4f}s" if avg_ttft is not None else "Average TTFT: N/A")
    print(f"Average Latency: {avg_latency:.4f}s")
    print(f"Average Throughput: {avg_throughput:.2f} tokens/sec")
    print(f"Total Cost: ${total_cost:.6f}")
    
    return summary

def compare_models(model_a_summary, model_b_summary, mmlu_comparison=None):
    """Print a side-by-side comparison of two models."""
    model_a = model_a_summary["model"]
    model_b = model_b_summary["model"]
    provider_a = model_a_summary.get("provider", "unknown")
    provider_b = model_b_summary.get("provider", "unknown")
    
    print(f"\n============ COMPARISON: {model_a} ({provider_a}) vs {model_b} ({provider_b}) ============")
    
    # Check if either model had fatal errors
    model_a_failed = model_a_summary["avg_latency"] is None
    model_b_failed = model_b_summary["avg_latency"] is None
    
    if model_a_failed and model_b_failed:
        print("\nBoth models failed to produce any results.")
        return
    elif model_a_failed:
        print(f"\n{model_a} failed to produce any results. Cannot compare.")
        return
    elif model_b_failed:
        print(f"\n{model_b} failed to produce any results. Cannot compare.")
        return
    
    # Provider Information
    print(f"\nProvider Information:")
    print(f"Model A: {provider_a}")
    print(f"Model B: {provider_b}")
    if provider_a != provider_b:
        print("*** Cross-platform comparison: Decentralized vs Centralized AI ***")
    
    # Speed Metrics section
    print("\nSpeed Metrics:")
    if model_a_summary["avg_ttft"] and model_b_summary["avg_ttft"]:
        ttft_ms_a = model_a_summary["avg_ttft"] * 1000  # Convert to ms
        ttft_ms_b = model_b_summary["avg_ttft"] * 1000  # Convert to ms
        print(f"Time to first token: {ttft_ms_a:.0f}ms vs {ttft_ms_b:.0f}ms")
    
    print(f"Total latency: {model_a_summary['avg_latency']:.1f}s vs {model_b_summary['avg_latency']:.1f}s")
    print(f"Tokens/sec: {model_a_summary['avg_throughput']:.0f} vs {model_b_summary['avg_throughput']:.0f}")
    
    # Accuracy Metrics section
    print("\nAccuracy Metrics:")
    if model_a_summary["accuracy_score"] is not None and model_b_summary["accuracy_score"] is not None:
        acc_a = model_a_summary["accuracy_score"] * 100
        acc_b = model_b_summary["accuracy_score"] * 100
        print(f"Standard Prompts: {acc_a:.1f}% vs {acc_b:.1f}%")
    
    # Add MMLU results if available
    if mmlu_comparison:
        acc_a_mmlu = mmlu_comparison["model_a_result"]["accuracy"] * 100
        acc_b_mmlu = mmlu_comparison["model_b_result"]["accuracy"] * 100
        print(f"MMLU Accuracy: {acc_a_mmlu:.1f}% vs {acc_b_mmlu:.1f}%")
    
    # Cost Analysis section
    print("\nCost Analysis:")
    print(f"Input tokens: ${model_a_summary['pricing']['input_rate']}/1K vs ${model_b_summary['pricing']['input_rate']}/1K")
    print(f"Output tokens: ${model_a_summary['pricing']['output_rate']}/1K vs ${model_b_summary['pricing']['output_rate']}/1K")
    
    cost_ratio_a = 1.0  # baseline
    cost_ratio_b = model_b_summary["total_cost"] / model_a_summary["total_cost"] if model_a_summary["total_cost"] else 0
    print(f"Cost-performance ratio: {cost_ratio_a:.1f}x vs {cost_ratio_b:.1f}x")
    
    # Add MMLU cost if available
    if mmlu_comparison:
        mmlu_cost_a = mmlu_comparison["model_a_result"]["total_cost"]
        mmlu_cost_b = mmlu_comparison["model_b_result"]["total_cost"]
        print(f"MMLU total cost: ${mmlu_cost_a:.6f} vs ${mmlu_cost_b:.6f}")
    
    # Decentralization insights (if comparing different providers)
    if provider_a != provider_b:
        print(f"\nDecentralization Analysis:")
        if provider_a == "lilypad" or provider_b == "lilypad":
            lilypad_model = model_a if provider_a == "lilypad" else model_b
            centralized_model = model_b if provider_a == "lilypad" else model_a
            print(f"• {lilypad_model} runs on decentralized compute network")
            print(f"• {centralized_model} runs on centralized infrastructure")
            print("• Lilypad provides cryptographic verification of compute execution")
            print("• Potential benefits: censorship resistance, geographic distribution")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark and compare Hyperbolic-hosted language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models with default test prompts
  hypercompare meta-llama/Meta-Llama-3-70B-Instruct meta-llama/Meta-Llama-3.1-8B-Instruct
        
  # Use custom prompts from a file
  hypercompare model1 model2 --prompts your_prompts.txt
        
Prompt file format (each line):
  "Who wrote Hamlet? | Shakespeare"  # Question with expected answer after |
  "What is the weather like today?"   # Question with no expected answer
"""
    )
    
    parser.add_argument("model_a", help="First model to benchmark")
    parser.add_argument("model_b", help="Second model to benchmark")
    parser.add_argument(
        "--providers", 
        nargs=2,
        choices=["hyperbolic", "lilypad"],
        help="Specify providers for model_a and model_b (e.g., --providers hyperbolic lilypad)"
    )
    parser.add_argument(
        "--prompts", 
        help="Path to a file containing test prompts. Each line is a prompt, optionally with expected answers after | character"
    )
    parser.add_argument(
        "--system", 
        default="You are a helpful assistant providing concise answers.",
        help="System prompt to use for all test cases"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show more detailed output and warnings"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature setting for model inference (default: 0 for deterministic outputs)"
    )
    parser.add_argument(
        "--skip-mmlu",
        action="store_true",
        help="Skip MMLU evaluation (faster but less comprehensive)"
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=0,
        help="Number of few-shot examples to use for MMLU (default: 0 for zero-shot)"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of questions per MMLU subject (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Determine providers for each model
    if args.providers:
        provider_a, provider_b = args.providers
    else:
        # Auto-detect providers based on model names
        provider_a = MODEL_PRICING.get(args.model_a, MODEL_PRICING["default"]).get("provider", "hyperbolic")
        provider_b = MODEL_PRICING.get(args.model_b, MODEL_PRICING["default"]).get("provider", "hyperbolic")
    
    # Validate model-provider combinations
    if provider_a == "lilypad" and args.model_a not in PROVIDER_MODELS["lilypad"]:
        print(f"Warning: Model '{args.model_a}' may not be available on Lilypad. Available models: {PROVIDER_MODELS['lilypad']}")
    if provider_b == "lilypad" and args.model_b not in PROVIDER_MODELS["lilypad"]:
        print(f"Warning: Model '{args.model_b}' may not be available on Lilypad. Available models: {PROVIDER_MODELS['lilypad']}")
    
    # Check for required environment variables
    required_keys = set()
    if provider_a == "lilypad" or provider_b == "lilypad":
        required_keys.add("LILYPAD_API_KEY")
    if provider_a == "hyperbolic" or provider_b == "hyperbolic":
        required_keys.add("HYPERBOLIC_API_KEY")
    
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        print(f"Error: The following API keys are required but not found in environment variables: {', '.join(missing_keys)}")
        print("Please create a .env file with your API keys or set them in your environment.")
        if "LILYPAD_API_KEY" in missing_keys:
            print("For Lilypad: Add LILYPAD_API_KEY to your .env file")
        if "HYPERBOLIC_API_KEY" in missing_keys:
            print("For Hyperbolic: Add HYPERBOLIC_API_KEY to your .env file")
        sys.exit(1)
    
    # Load test prompts
    test_prompts = DEFAULT_TEST_PROMPTS
    if args.prompts:
        try:
            test_prompts = parse_prompts_file(args.prompts)
            if not test_prompts:
                print(f"Warning: No valid prompts found in {args.prompts}, using default prompts")
                test_prompts = DEFAULT_TEST_PROMPTS
        except Exception as e:
            print(f"Error reading prompts file: {e}")
            print("Using default test prompts instead.")
    
    # Initialize API clients based on providers
    try:
        client_a = get_api_client(args.model_a, provider_a)
        client_b = get_api_client(args.model_b, provider_b)
    except ValueError as e:
        print(f"Error initializing API clients: {e}")
        sys.exit(1)
    
    # Run standard tests for first model
    print("\n===== Running Standard Benchmark =====")
    model_a_summary = run_accuracy_tests(client_a, args.model_a, test_prompts, args.system, temperature=args.temperature, verbose=args.verbose, provider=provider_a)
    
    # Run standard tests for second model
    model_b_summary = run_accuracy_tests(client_b, args.model_b, test_prompts, args.system, temperature=args.temperature, verbose=args.verbose, provider=provider_b)
    
    # Run MMLU evaluation if available and not skipped
    mmlu_comparison = None
    if MMLU_AVAILABLE and not args.skip_mmlu:
        try:
            print("\n===== Running MMLU Evaluation =====")
            # Load MMLU dataset
            print("Loading MMLU dataset...")
            subjects = DEFAULT_SUBJECTS
            
            datasets = []
            for subject in subjects:
                try:
                    ds = load_dataset("cais/mmlu", subject, split="test")
                    # Limit the number of questions per subject
                    if args.num_questions < len(ds):
                        ds = ds.select(range(args.num_questions))
                    datasets.append(ds)
                    print(f"Loaded {len(ds)} questions for {subject}")
                except Exception as e:
                    print(f"Error loading {subject}: {e}")
            
            from datasets import concatenate_datasets
            if not datasets:
                print("No datasets could be loaded. Skipping MMLU evaluation.")
            else:
                full_dataset = concatenate_datasets(datasets)
                
                # Prepare few-shot examples if n_shots > 0
                few_shot_examples = None
                if args.n_shots > 0:
                    actual_n_shots = min(args.n_shots, len(full_dataset))
                    
                    if actual_n_shots < len(full_dataset):
                        # Use the first n_shots examples for few-shot prompting
                        few_shot_dataset = full_dataset.select(range(actual_n_shots))
                        few_shot_examples = prepare_few_shot_examples(few_shot_dataset, actual_n_shots)
                        # Use the remaining examples for testing
                        test_dataset = full_dataset.select(range(actual_n_shots, len(full_dataset)))
                    else:
                        # Not enough data for separate few-shot examples and test cases
                        # Use the same examples for both (not ideal but allows testing)
                        print(f"Warning: Dataset only has {len(full_dataset)} examples, using same examples for few-shot and testing")
                        few_shot_dataset = full_dataset.select(range(actual_n_shots))
                        few_shot_examples = prepare_few_shot_examples(few_shot_dataset, actual_n_shots)
                        test_dataset = full_dataset
                else:
                    test_dataset = full_dataset
                
                test_cases = prepare_mmlu_test_cases(test_dataset, num_questions=None)  # Use all loaded questions
                
                # Evaluate model A
                print(f"\nEvaluating {args.model_a}...")
                result_a = evaluate_model_on_mmlu(client_a, args.model_a, test_cases, 
                                                system_prompt=args.system, 
                                                few_shot_examples=few_shot_examples,
                                                temperature=args.temperature, 
                                                verbose=args.verbose)
                
                # Evaluate model B
                print(f"\nEvaluating {args.model_b}...")
                result_b = evaluate_model_on_mmlu(client_b, args.model_b, test_cases, 
                                                system_prompt=args.system, 
                                                few_shot_examples=few_shot_examples,
                                                temperature=args.temperature, 
                                                verbose=args.verbose)
                
                # Compare results
                comparison = compare_mmlu_results(result_a, result_b)
                mmlu_comparison = {
                    "model_a_result": result_a,
                    "model_b_result": result_b,
                    "comparison": comparison
                }
        except Exception as e:
            print(f"Error running MMLU evaluation: {e}")
            print("Continuing with standard benchmark results only.")
    
    # Compare the models with comprehensive results
    compare_models(model_a_summary, model_b_summary, mmlu_comparison)
    
if __name__ == "__main__":
    main() 