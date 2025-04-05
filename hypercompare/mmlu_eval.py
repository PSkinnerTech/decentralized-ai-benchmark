#!/usr/bin/env python3
"""
MMLU Evaluation Module

This module provides functions to evaluate language models using the MMLU benchmark.
It integrates with the hyperapi and mmlu_dataset modules to benchmark models
on multiple-choice questions across different subjects.
"""

import os
import time
import openai
from dotenv import load_dotenv
import argparse
from datasets import load_dataset

# Import our modules
from hyperapi import get_model_metrics
from mmlu_dataset import download_mmlu_dataset, prepare_mmlu_test_cases, prepare_few_shot_examples, DEFAULT_SUBJECTS

load_dotenv()

def evaluate_model_on_mmlu(client, model_name, test_cases, system_prompt=None, few_shot_examples=None, temperature=0, verbose=False):
    """
    Evaluate a model on MMLU questions using Hyperbolic API.
    
    Args:
        client: An initialized OpenAI client for Hyperbolic
        model_name: Name of the model to test
        test_cases: List of test cases from prepare_mmlu_test_cases
        system_prompt: Optional system prompt to use
        few_shot_examples: Optional list of (question, answer) tuples for few-shot learning
        temperature: Temperature setting for model inference (default: 0 for deterministic outputs)
        verbose: Whether to print additional information
        
    Returns:
        Dictionary containing performance metrics and accuracy scores
    """
    # Default system prompt if none is provided
    if system_prompt is None:
        system_prompt = """You are an expert at solving multiple-choice questions.
Answer with ONLY the letter (A, B, C, or D) corresponding to the correct answer.
Do not include any explanation, just the single letter.
For example, if you think option B is correct, respond with just 'B'.
Do not include any other text, punctuation, or explanations."""
    
    # Track metrics for each question
    results = []
    correct_answers = 0
    total_questions = len(test_cases)
    
    # Track subject-specific performance
    subject_results = {}
    
    print(f"Evaluating {model_name} on {total_questions} MMLU questions...")
    if few_shot_examples:
        print(f"Using {len(few_shot_examples)} few-shot examples")
    else:
        print("Using zero-shot evaluation")
    
    # Process each test case
    for i, test_case in enumerate(test_cases):
        subject = test_case.get("subject", "unknown")
        prompt = test_case["prompt"]
        expected_answer = test_case["expected_keywords"][0]  # Just the letter (A, B, C, D)
        
        # Set up the conversation
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add few-shot examples if provided
        if few_shot_examples:
            for example_prompt, example_answer in few_shot_examples:
                messages.append({"role": "user", "content": example_prompt})
                messages.append({"role": "assistant", "content": example_answer})
        
        # Add the current question
        messages.append({"role": "user", "content": prompt})
        
        print(f"\nQuestion {i+1}/{total_questions} [{subject}]:")
        print(f"Prompt: {prompt.split('Question: ')[1].split('Choices:')[0].strip()}")
        
        # Get metrics and response
        metrics = get_model_metrics(client, model_name, messages, temperature=temperature, verbose=verbose)
        
        # Check if we got a valid response
        if metrics.get("response_content"):
            response = metrics["response_content"].strip()
            
            # Extract just the letter from the response if it contains more text
            answer_letter = None
            
            # First try to extract just the first character if it's a letter
            if response and response[0].upper() in ["A", "B", "C", "D"]:
                answer_letter = response[0].upper()
            # Then check for any occurrence of "A", "B", "C", "D" in the response
            else:
                # Look for patterns like "Answer: A" or similar
                for pattern in ["ANSWER:", "ANSWER IS:", "THE ANSWER IS:", "CORRECT ANSWER:"]:
                    if pattern in response.upper():
                        parts = response.upper().split(pattern)
                        if len(parts) > 1 and parts[1].strip() and parts[1].strip()[0] in ["A", "B", "C", "D"]:
                            answer_letter = parts[1].strip()[0]
                            break
                
                # If still no answer found, look for any letter in the response
                if not answer_letter:
                    for letter in ["A", "B", "C", "D"]:
                        if letter in response.upper():
                            # Check if it's a standalone letter or surrounded by non-alphanumeric characters
                            pattern = r'\b' + letter + r'\b'
                            import re
                            if re.search(pattern, response.upper()):
                                answer_letter = letter
                                break
            
            is_correct = answer_letter == expected_answer
            
            if is_correct:
                correct_answers += 1
            
            # Store result for this question
            result = {
                "subject": subject,
                "question": test_case["original_question"],
                "expected_answer": expected_answer,
                "model_answer": answer_letter,
                "is_correct": is_correct,
                "metrics": metrics
            }
            
            # Add to subject-specific results
            if subject not in subject_results:
                subject_results[subject] = {"correct": 0, "total": 0}
            subject_results[subject]["total"] += 1
            if is_correct:
                subject_results[subject]["correct"] += 1
            
            results.append(result)
            
            # Print brief result
            print(f"Expected: {expected_answer}, Got: {answer_letter}, Correct: {is_correct}")
            print(f"TTFT: {metrics['ttft']:.4f}s, Latency: {metrics['latency']:.4f}s, Cost: ${metrics['cost']:.6f}")
        else:
            print(f"Error: {metrics.get('error', 'Unknown error')}")
            results.append({
                "subject": subject,
                "question": test_case["original_question"],
                "expected_answer": expected_answer,
                "model_answer": None,
                "is_correct": False,
                "metrics": metrics,
                "error": metrics.get("error")
            })
    
    # Calculate overall metrics
    total_latency = sum(r["metrics"]["latency"] for r in results if r["metrics"]["latency"] is not None)
    valid_ttft = [r["metrics"]["ttft"] for r in results if r["metrics"]["ttft"] is not None]
    avg_ttft = sum(valid_ttft) / len(valid_ttft) if valid_ttft else None
    avg_latency = total_latency / total_questions if total_questions > 0 else 0
    avg_throughput = sum(r["metrics"]["throughput"] for r in results) / total_questions if total_questions > 0 else 0
    total_cost = sum(r["metrics"]["cost"] for r in results)
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    # Calculate subject-specific accuracy
    for subject in subject_results:
        subject_results[subject]["accuracy"] = (
            subject_results[subject]["correct"] / subject_results[subject]["total"]
            if subject_results[subject]["total"] > 0 else 0
        )
    
    # Print summary
    print("\n--- MMLU Evaluation Summary ---")
    print(f"Model: {model_name}")
    print(f"Questions: {total_questions}")
    print(f"Correct: {correct_answers}")
    print(f"Accuracy: {accuracy*100:.1f}% ({correct_answers}/{total_questions})")
    print(f"Average TTFT: {avg_ttft:.4f}s" if avg_ttft else "Average TTFT: N/A")
    print(f"Average Latency: {avg_latency:.4f}s")
    print(f"Average Throughput: {avg_throughput:.2f} tokens/sec")
    print(f"Total Cost: ${total_cost:.6f}")
    
    # Print subject-specific results
    print("\n--- Subject Breakdown ---")
    for subject, data in sorted(subject_results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        print(f"{subject}: {data['accuracy']*100:.1f}% ({data['correct']}/{data['total']})")
    
    return {
        "model": model_name,
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": accuracy,
        "avg_ttft": avg_ttft,
        "avg_latency": avg_latency,
        "avg_throughput": avg_throughput,
        "total_cost": total_cost,
        "subject_results": subject_results,
        "results": results
    }

def compare_mmlu_results(result_a, result_b):
    """
    Compare MMLU results between two models.
    
    Args:
        result_a: Results dictionary for model A
        result_b: Results dictionary for model B
        
    Returns:
        Dictionary containing comparison metrics
    """
    model_a = result_a["model"]
    model_b = result_b["model"]
    
    # Print comparison
    print("\n--- MMLU Comparison ---")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    
    # Compare accuracy
    print("\nAccuracy Metrics:")
    print(f"MMLU score: {result_a['accuracy']*100:.1f}% vs {result_b['accuracy']*100:.1f}%")
    
    # Compare speed
    print("\nSpeed Metrics:")
    if result_a["avg_ttft"] and result_b["avg_ttft"]:
        ttft_ms_a = result_a["avg_ttft"] * 1000
        ttft_ms_b = result_b["avg_ttft"] * 1000
        print(f"Time to first token: {ttft_ms_a:.0f}ms vs {ttft_ms_b:.0f}ms")
    
    print(f"Total latency: {result_a['avg_latency']:.1f}s vs {result_b['avg_latency']:.1f}s")
    print(f"Tokens/sec: {result_a['avg_throughput']:.0f} vs {result_b['avg_throughput']:.0f}")
    
    # Compare cost
    print("\nCost Analysis:")
    print(f"Total cost: ${result_a['total_cost']:.6f} vs ${result_b['total_cost']:.6f}")
    
    # Cost-performance ratio (cost per correct answer)
    if result_a['correct_answers'] > 0 and result_b['correct_answers'] > 0:
        cost_per_correct_a = result_a['total_cost'] / result_a['correct_answers']
        cost_per_correct_b = result_b['total_cost'] / result_b['correct_answers']
        
        ratio_a = 1.0  # baseline
        ratio_b = cost_per_correct_b / cost_per_correct_a
        print(f"Cost-performance ratio: {ratio_a:.1f}x vs {ratio_b:.1f}x")
    else:
        if result_a['correct_answers'] == 0 and result_b['correct_answers'] == 0:
            print("Cost-performance ratio: N/A (both models got 0 correct answers)")
        elif result_a['correct_answers'] == 0:
            print("Cost-performance ratio: N/A vs 1.0x (model A got 0 correct answers)")
        else:  # result_b['correct_answers'] == 0
            print("Cost-performance ratio: 1.0x vs N/A (model B got 0 correct answers)")
    
    return {
        "model_a": model_a,
        "model_b": model_b,
        "accuracy_diff": result_b["accuracy"] - result_a["accuracy"],
        "ttft_diff": result_b["avg_ttft"] - result_a["avg_ttft"] if result_a["avg_ttft"] and result_b["avg_ttft"] else None,
        "latency_diff": result_b["avg_latency"] - result_a["avg_latency"],
        "throughput_diff": result_b["avg_throughput"] - result_a["avg_throughput"],
        "cost_diff": result_b["total_cost"] - result_a["total_cost"]
    }

def main():
    """Main CLI entry point for MMLU evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate LLMs on MMLU benchmark")
    parser.add_argument("model_a", help="First model to benchmark")
    parser.add_argument("model_b", help="Second model to benchmark")
    parser.add_argument("--subjects", nargs="+", help="Specific MMLU subjects to test")
    parser.add_argument("--num_questions", type=int, default=5, help="Number of questions per subject")
    parser.add_argument("--system", help="Custom system prompt to use")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--temperature", type=float, default=0, 
                        help="Temperature setting for model inference (default: 0 for deterministic outputs)")
    parser.add_argument("--split", default="test", choices=["dev", "validation", "test"],
                       help="Dataset split to use")
    parser.add_argument("--n_shots", type=int, default=0, 
                       help="Number of few-shot examples to use (default: 0 for zero-shot)")
    
    args = parser.parse_args()
    
    # Initialize Hyperbolic client
    client = openai.OpenAI(
        api_key=os.getenv('HYPERBOLIC_API_KEY'),
        base_url="https://api.hyperbolic.xyz/v1",
    )
    
    # Load MMLU dataset
    print("Loading MMLU dataset...")
    subjects = args.subjects if args.subjects else DEFAULT_SUBJECTS
    
    datasets = []
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split=args.split)
            # Limit the number of questions per subject
            if args.num_questions < len(ds):
                ds = ds.select(range(args.num_questions))
            datasets.append(ds)
            print(f"Loaded {len(ds)} questions for {subject}")
        except Exception as e:
            print(f"Error loading {subject}: {e}")
    
    from datasets import concatenate_datasets
    if not datasets:
        print("No datasets could be loaded. Exiting.")
        return
    
    full_dataset = concatenate_datasets(datasets)
    
    # Prepare few-shot examples if n_shots > 0
    few_shot_examples = None
    if args.n_shots > 0:
        # Make sure we don't try to use more shots than we have examples
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
    result_a = evaluate_model_on_mmlu(client, args.model_a, test_cases, 
                                    system_prompt=args.system, 
                                    few_shot_examples=few_shot_examples,
                                    temperature=args.temperature, 
                                    verbose=args.verbose)
    
    # Evaluate model B
    print(f"\nEvaluating {args.model_b}...")
    result_b = evaluate_model_on_mmlu(client, args.model_b, test_cases, 
                                     system_prompt=args.system, 
                                     few_shot_examples=few_shot_examples,
                                     temperature=args.temperature, 
                                     verbose=args.verbose)
    
    # Compare results
    compare_mmlu_results(result_a, result_b)

if __name__ == "__main__":
    main() 