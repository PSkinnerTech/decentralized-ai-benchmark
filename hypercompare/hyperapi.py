import os
import openai
import time
from dotenv import load_dotenv

load_dotenv()

# Simplified model pricing (cents per 1K tokens)
# This would ideally come from a configuration file or API
MODEL_PRICING = {
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "input_rate": 0.0025,  # $0.0025 per 1K input tokens
        "output_rate": 0.0035,  # $0.0035 per 1K output tokens
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "input_rate": 0.0028,  # $0.0028 per 1K input tokens
        "output_rate": 0.0038,  # $0.0038 per 1K output tokens
    },
    # Add more models as needed
    # Default fallback for unknown models
    "default": {
        "input_rate": 0.003,
        "output_rate": 0.004,
    }
}

def get_model_metrics(client, model_name, messages):
    """Calls the Hyperbolic API, calculates performance metrics, and returns them."""
    start_time = time.time()
    first_token_time = None
    response_content = ""
    completion_tokens = 0
    prompt_tokens = 0
    last_chunk = None

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7, # Keep parameters consistent or pass them in
            max_tokens=1024,
            stream=True,
        )

        for chunk in stream:
            if first_token_time is None and chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                first_token_time = time.time()
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                response_content += chunk.choices[0].delta.content
            last_chunk = chunk

        end_time = time.time()

        # Extract usage from the last chunk
        if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
            prompt_tokens = last_chunk.usage.prompt_tokens
            completion_tokens = last_chunk.usage.completion_tokens
        else:
             print("Warning: Could not extract usage data from the last stream chunk.")

        # Calculate metrics
        latency = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        throughput = completion_tokens / latency if latency > 0 and completion_tokens > 0 else 0

        # Calculate cost
        # Get pricing rates for this model or use default
        pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["default"])
        input_rate = pricing["input_rate"]
        output_rate = pricing["output_rate"]
        
        # Apply the cost formula from the guide
        cost = ((prompt_tokens * input_rate) + (completion_tokens * output_rate)) / 1000
        
        return {
            "response_content": response_content,
            "ttft": ttft,
            "latency": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "throughput": throughput,
            "cost": cost,
            "input_rate": input_rate,
            "output_rate": output_rate
        }

    except Exception as e:
        print(f"Error during API call or metric calculation: {e}")
        return {
            "response_content": None,
            "ttft": None,
            "latency": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "throughput": 0,
            "cost": 0,
            "error": str(e)
        }


if __name__ == "__main__":
    # --- Configuration --- 
    model_to_test = "meta-llama/Meta-Llama-3-70B-Instruct"
    # System prompt for accuracy tests (can be simple)
    accuracy_system_prompt = "You are a helpful assistant providing concise answers."
    
    # Accuracy Test Data (prompt and expected keywords)
    accuracy_test_prompts = [
        {"prompt": "Who wrote Hamlet?", "expected_keywords": ["Shakespeare"]},
        {"prompt": "What is 2 + 2?", "expected_keywords": ["4", "four"]},
        {"prompt": "What is the capital of France?", "expected_keywords": ["Paris"]},
        # Add more test cases here if needed
    ]
    # ---------------------

    # Initialize OpenAI client for Hyperbolic
    hyperbolic_client = openai.OpenAI(
        api_key=os.getenv('HYPERBOLIC_API_KEY'),
        base_url="https://api.hyperbolic.xyz/v1",
    )

    print(f"--- Testing Model Accuracy: {model_to_test} ---")
    correct_responses = 0
    total_prompts = len(accuracy_test_prompts)
    total_cost = 0

    for i, test_case in enumerate(accuracy_test_prompts):
        print(f"\nRunning Test {i+1}/{total_prompts}: '{test_case['prompt']}'")
        messages = [
            {"role": "system", "content": accuracy_system_prompt},
            {"role": "user", "content": test_case["prompt"]},
        ]
        
        metrics = get_model_metrics(hyperbolic_client, model_to_test, messages)
        total_cost += metrics.get("cost", 0)

        is_correct = False
        if metrics.get("response_content"):
            # Print detailed metrics for this test case
            print(f"  TTFT: {metrics['ttft']:.4f}s" if metrics['ttft'] is not None else "  TTFT: N/A")
            print(f"  Latency: {metrics['latency']:.4f}s")
            print(f"  Prompt Tokens: {metrics['prompt_tokens']}")
            print(f"  Completion Tokens: {metrics['completion_tokens']}")
            print(f"  Throughput: {metrics['throughput']:.2f} tokens/sec")
            print(f"  Cost: ${metrics['cost']:.6f} (${metrics['input_rate']}/1K input, ${metrics['output_rate']}/1K output)")
            
            response_lower = metrics["response_content"].lower()
            print(f"  Response Snippet: {metrics['response_content'][:80]}...") # Shorter snippet
            for keyword in test_case["expected_keywords"]:
                if keyword.lower() in response_lower:
                    is_correct = True
                    print(f"  Result: Correct (found '{keyword}')")
                    correct_responses += 1
                    break # Stop checking keywords for this prompt once one match is found
            if not is_correct:
                 print(f"  Result: Incorrect (expected keywords not found: {test_case['expected_keywords']})")
        else:
            print(f"  Result: Error - No response content found. Error: {metrics.get('error')}")

    # --- Accuracy Summary ---
    accuracy_score = correct_responses / total_prompts if total_prompts > 0 else 0
    print("\n--- Accuracy Summary ---")
    print(f"Model: {model_to_test}")
    print(f"Correct Responses: {correct_responses}/{total_prompts}")
    print(f"Accuracy Score: {accuracy_score:.2f}")
    print(f"Total Cost: ${total_cost:.6f}")

    # Note: We removed the original single test run. 
    # The CLI will handle running specific prompts vs accuracy benchmarks. 