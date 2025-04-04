#!/usr/bin/env python3
import os
import sys
import argparse
import openai
from dotenv import load_dotenv
import time

# Import our metrics function
from hyperapi import get_model_metrics, MODEL_PRICING

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

def run_accuracy_tests(client, model_name, test_prompts, system_prompt, verbose=False):
    """Run accuracy tests on a model and return metrics."""
    print(f"\n--- Testing Model: {model_name} ---")
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
        
        metrics = get_model_metrics(client, model_name, messages, verbose=verbose)
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
    if has_expected_answers:
        print(f"Correct Responses: {correct_responses}/{total_prompts}")
        print(f"Accuracy Score: {accuracy_score:.2f}")
    print(f"Average TTFT: {avg_ttft:.4f}s" if avg_ttft is not None else "Average TTFT: N/A")
    print(f"Average Latency: {avg_latency:.4f}s")
    print(f"Average Throughput: {avg_throughput:.2f} tokens/sec")
    print(f"Total Cost: ${total_cost:.6f}")
    
    return summary

def compare_models(model_a_summary, model_b_summary):
    """Print a side-by-side comparison of two models."""
    model_a = model_a_summary["model"]
    model_b = model_b_summary["model"]
    
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
    
    # Speed Metrics section
    print("\nSpeed Metrics:")
    if model_a_summary["avg_ttft"] and model_b_summary["avg_ttft"]:
        ttft_ms_a = model_a_summary["avg_ttft"] * 1000  # Convert to ms
        ttft_ms_b = model_b_summary["avg_ttft"] * 1000  # Convert to ms
        print(f"Time to first token: {ttft_ms_a:.0f}ms vs {ttft_ms_b:.0f}ms")
    
    print(f"Total latency: {model_a_summary['avg_latency']:.1f}s vs {model_b_summary['avg_latency']:.1f}s")
    print(f"Tokens/sec: {model_a_summary['avg_throughput']:.0f} vs {model_b_summary['avg_throughput']:.0f}")
    
    # Accuracy Metrics section
    if model_a_summary["accuracy_score"] is not None and model_b_summary["accuracy_score"] is not None:
        print("\nAccuracy Metrics:")
        acc_a = model_a_summary["accuracy_score"] * 100
        acc_b = model_b_summary["accuracy_score"] * 100
        print(f"Accuracy: {acc_a:.1f}% vs {acc_b:.1f}%")
        
        # Note: We don't have MMLU, HumanEval, or Consistency metrics in our implementation
        # but we can display the appropriate accuracy metrics we do have
    
    # Cost Analysis section
    print("\nCost Analysis:")
    print(f"Input tokens: ${model_a_summary['pricing']['input_rate']}/1K vs ${model_b_summary['pricing']['input_rate']}/1K")
    print(f"Output tokens: ${model_a_summary['pricing']['output_rate']}/1K vs ${model_b_summary['pricing']['output_rate']}/1K")
    
    cost_ratio_a = 1.0  # baseline
    cost_ratio_b = model_b_summary["total_cost"] / model_a_summary["total_cost"] if model_a_summary["total_cost"] else 0
    print(f"Cost-performance ratio: {cost_ratio_a:.1f}x vs {cost_ratio_b:.1f}x")

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
    
    args = parser.parse_args()
    
    # Check for environment variables
    if not os.getenv("HYPERBOLIC_API_KEY"):
        print("Error: HYPERBOLIC_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key or set it in your environment.")
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
    
    # Initialize OpenAI client for Hyperbolic
    hyperbolic_client = openai.OpenAI(
        api_key=os.getenv('HYPERBOLIC_API_KEY'),
        base_url="https://api.hyperbolic.xyz/v1",
    )
    
    # Run tests for first model
    model_a_summary = run_accuracy_tests(hyperbolic_client, args.model_a, test_prompts, args.system, verbose=args.verbose)
    
    # Run tests for second model
    model_b_summary = run_accuracy_tests(hyperbolic_client, args.model_b, test_prompts, args.system, verbose=args.verbose)
    
    # Compare the models
    compare_models(model_a_summary, model_b_summary)
    
if __name__ == "__main__":
    main() 