#!/usr/bin/env python3
"""
MMLU Dataset Downloader

This script downloads and explores the MMLU (Massive Multitask Language Understanding) 
dataset from Hugging Face, which can be used for benchmarking language models.
"""

from datasets import load_dataset
import argparse

# Default subjects to use if none specified
DEFAULT_SUBJECTS = [
    'high_school_mathematics', 
    'high_school_computer_science',
    'elementary_mathematics',
    'college_computer_science',
    'college_mathematics'
]

def download_mmlu_dataset(split="test", subject=None, subjects=None, limit=5):
    """
    Download the MMLU dataset from Hugging Face.
    
    Args:
        split (str): The dataset split to load ('dev', 'validation', or 'test')
        subject (str, optional): Single specific subject to load
        subjects (list, optional): List of subjects to load
        limit (int): Number of examples to display
        
    Returns:
        dataset: HuggingFace dataset object
    """
    try:
        # Handle the three possible input cases
        if subject:
            # Load a single subject
            print(f"Loading MMLU dataset for subject: {subject}")
            dataset = load_dataset("cais/mmlu", subject, split=split)
        elif subjects:
            # Load multiple specified subjects
            print(f"Loading MMLU dataset for subjects: {', '.join(subjects)}")
            datasets = []
            for subj in subjects:
                print(f"  Loading {subj}...")
                ds = load_dataset("cais/mmlu", subj, split=split)
                datasets.append(ds)
            # Concatenate all datasets
            from datasets import concatenate_datasets
            dataset = concatenate_datasets(datasets)
        else:
            # Load default subjects
            print(f"Loading default MMLU subjects: {', '.join(DEFAULT_SUBJECTS)}")
            datasets = []
            for subj in DEFAULT_SUBJECTS:
                print(f"  Loading {subj}...")
                try:
                    ds = load_dataset("cais/mmlu", subj, split=split)
                    datasets.append(ds)
                except Exception as e:
                    print(f"  Error loading {subj}: {e}")
            # Concatenate all datasets
            from datasets import concatenate_datasets
            dataset = concatenate_datasets(datasets)
        
        # Print some basic information about the dataset
        print(f"Dataset size: {len(dataset)} examples")
        features = list(dataset.features.keys())
        print(f"Dataset features: {features}")
        
        # Display some examples
        print(f"\nDisplaying {limit} example(s):")
        for i in range(min(limit, len(dataset))):
            example = dataset[i]
            print(f"\nExample {i+1}:")
            print(f"Question: {example['question']}")
            print(f"Choices: {example['choices']}")
            print(f"Answer: {example['answer']}")
        
        return dataset
    
    except Exception as e:
        print(f"Error downloading MMLU dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def format_mmlu_prompt(question, choices):
    """
    Format an MMLU question for LLM evaluation.
    
    Args:
        question (str): The question text
        choices (list): List of possible answers
        
    Returns:
        str: Formatted prompt for the model
    """
    prompt = f"Question: {question}\n\nChoices:\n"
    
    # Add letter options for each choice
    for i, choice in enumerate(choices):
        letter = chr(65 + i)  # A, B, C, D
        prompt += f"{letter}. {choice}\n"
    
    prompt += "\nAnswer with the letter of the correct choice."
    
    return prompt

def prepare_mmlu_test_cases(dataset, num_questions=10):
    """
    Prepare MMLU test cases for model evaluation.
    
    Args:
        dataset: HuggingFace dataset containing MMLU questions
        num_questions (int): Number of questions to use (if None, use all)
        
    Returns:
        list: List of test cases with prompts and expected answers
    """
    test_cases = []
    
    # Use all questions if num_questions is None or larger than dataset
    question_count = min(num_questions, len(dataset)) if num_questions else len(dataset)
    
    for i in range(question_count):
        example = dataset[i]
        question = example['question']
        choices = example['choices']
        
        # The answer is a numeric index (0-3) but we need to convert to letter (A-D)
        correct_answer_index = example['answer']
        correct_answer_letter = chr(65 + correct_answer_index)  # Convert to A, B, C, D
        
        # Format the prompt
        prompt = format_mmlu_prompt(question, choices)
        
        test_case = {
            "prompt": prompt,
            "expected_keywords": [correct_answer_letter],
            "subject": example.get('subject', 'unknown'),
            "original_question": question,
            "choices": choices,
            "correct_index": correct_answer_index
        }
        
        test_cases.append(test_case)
    
    return test_cases

def prepare_few_shot_examples(dataset, n_shots=5):
    """
    Create few-shot examples for MMLU evaluation.
    
    Args:
        dataset: HuggingFace dataset containing MMLU questions
        n_shots: Number of examples to use (default: 5)
        
    Returns:
        list: List of (question, answer) tuples for few-shot prompting
    """
    examples = []
    # Use the first n_shots examples from the dataset
    # We'll use different examples than the test cases
    n_examples = min(n_shots, len(dataset))
    
    for i in range(n_examples):
        example = dataset[i]
        question = example['question']
        choices = example['choices']
        
        # Format the prompt
        prompt = format_mmlu_prompt(question, choices)
        
        # The answer is a numeric index (0-3) that we convert to letter (A-D)
        correct_answer_index = example['answer']
        correct_answer_letter = chr(65 + correct_answer_index)  # Convert to A, B, C, D
        
        examples.append((prompt, correct_answer_letter))
    
    return examples

def list_available_subjects():
    """List all available subjects in the MMLU dataset."""
    try:
        # Use the dataset info to get available configs (subjects)
        info = load_dataset("cais/mmlu", "abstract_algebra", split="test")
        
        # This list is taken from the dataset documentation
        available_subjects = [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 
            'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 
            'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 
            'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 
            'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
            'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 
            'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 
            'high_school_physics', 'high_school_psychology', 'high_school_statistics', 
            'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 
            'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 
            'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 
            'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 
            'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 
            'sociology', 'us_foreign_policy', 'virology', 'world_religions'
        ]
        
        print("Available MMLU subjects:")
        for subject in sorted(available_subjects):
            print(f"- {subject}")
        
    except Exception as e:
        print(f"Error listing subjects: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download and explore the MMLU dataset")
    parser.add_argument("--subject", help="Specific subject to load")
    parser.add_argument("--subjects", nargs="+", help="Multiple subjects to load")
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to display")
    parser.add_argument("--split", default="test", choices=["dev", "validation", "test"], 
                        help="Dataset split to load")
    parser.add_argument("--list-subjects", action="store_true", help="List all available subjects")
    parser.add_argument("--format-example", action="store_true", help="Show a formatted example prompt")
    
    args = parser.parse_args()
    
    if args.list_subjects:
        list_available_subjects()
    elif args.format_example:
        # Load a single example and show formatted prompt
        dataset = load_dataset("cais/mmlu", "high_school_mathematics", split="test").select(range(1))
        test_cases = prepare_mmlu_test_cases(dataset)
        print("\nFormulated prompt for model evaluation:")
        print("-" * 50)
        print(test_cases[0]["prompt"])
        print("-" * 50)
        print(f"Expected answer: {test_cases[0]['expected_keywords'][0]}")
    else:
        dataset = download_mmlu_dataset(split=args.split, subject=args.subject, subjects=args.subjects, limit=args.limit)
        
        if dataset and args.limit > 0:
            # Show an example of a formatted prompt
            test_cases = prepare_mmlu_test_cases(dataset, num_questions=1)
            print("\nFormulated prompt for model evaluation:")
            print("-" * 50)
            print(test_cases[0]["prompt"])
            print("-" * 50)
            print(f"Expected answer: {test_cases[0]['expected_keywords'][0]}")

if __name__ == "__main__":
    main() 