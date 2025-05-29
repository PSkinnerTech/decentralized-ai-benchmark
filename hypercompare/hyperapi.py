import os
import openai
import time
from dotenv import load_dotenv

"""
Core API module for the Hyperbolic model benchmarking tool.
Provides functions to call the Hyperbolic API and calculate performance metrics.
Now supports both Hyperbolic and Lilypad providers for comparison.
"""

load_dotenv()

# Model pricing with provider information (cents per 1K tokens)
MODEL_PRICING = {
    # Hyperbolic models
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "input_rate": 0.0025,  # $0.0025 per 1K input tokens
        "output_rate": 0.0035,  # $0.0035 per 1K output tokens
        "provider": "hyperbolic"
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "input_rate": 0.002,   # $0.002 per 1K input tokens
        "output_rate": 0.003,  # $0.003 per 1K output tokens
        "provider": "hyperbolic"
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "input_rate": 0.0028,  # $0.0028 per 1K input tokens
        "output_rate": 0.0038,  # $0.0038 per 1K output tokens
        "provider": "hyperbolic"
    },
    
    # Lilypad models (pricing TBD - contact Lilypad team)
    "deepseek-r1:7b": {
        "input_rate": 0.001,   # TBD - contact Lilypad team
        "output_rate": 0.002,  # TBD - contact Lilypad team
        "provider": "lilypad"
    },
    "llama3.1:8b": {
        "input_rate": 0.001,   # TBD
        "output_rate": 0.002,  # TBD
        "provider": "lilypad"
    },
    "qwen2.5:7b": {
        "input_rate": 0.001,   # TBD
        "output_rate": 0.002,  # TBD
        "provider": "lilypad"
    },
    "qwen2.5-coder:7b": {
        "input_rate": 0.001,   # TBD
        "output_rate": 0.002,  # TBD
        "provider": "lilypad"
    },
    "phi4-mini:3.8b": {
        "input_rate": 0.0008,  # TBD
        "output_rate": 0.0015, # TBD
        "provider": "lilypad"
    },
    "mistral:7b": {
        "input_rate": 0.001,   # TBD
        "output_rate": 0.002,  # TBD
        "provider": "lilypad"
    },
    "llava:7b": {
        "input_rate": 0.001,   # TBD - Vision model
        "output_rate": 0.002,  # TBD
        "provider": "lilypad"
    },
    
    # Default fallback for unknown models
    "default": {
        "input_rate": 0.003,
        "output_rate": 0.004,
        "provider": "hyperbolic"
    }
}

# Provider model availability matrix
PROVIDER_MODELS = {
    "hyperbolic": [
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ],
    "lilypad": [
        "deepseek-r1:7b",
        "llama3.1:8b", 
        "qwen2.5:7b",
        "qwen2.5-coder:7b",
        "phi4-mini:3.8b",
        "mistral:7b",
        "llava:7b"
    ]
}

def get_api_client(model_name, provider=None):
    """Return appropriate API client based on model or explicit provider."""
    if provider == "lilypad" or MODEL_PRICING.get(model_name, {}).get("provider") == "lilypad":
        api_key = os.getenv('LILYPAD_API_KEY')
        if not api_key:
            raise ValueError("LILYPAD_API_KEY not found in environment variables. Please add it to your .env file.")
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://anura-testnet.lilypad.tech/api/v1",
        )
    else:  # Default to Hyperbolic
        api_key = os.getenv('HYPERBOLIC_API_KEY')
        if not api_key:
            raise ValueError("HYPERBOLIC_API_KEY not found in environment variables. Please add it to your .env file.")
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://api.hyperbolic.xyz/v1",
        )

def get_model_metrics(client, model_name, messages, temperature=0, verbose=False, provider=None):
    """
    Calls the API (Hyperbolic or Lilypad), calculates performance metrics, and returns them.
    
    Args:
        client: An initialized OpenAI client for the appropriate provider
        model_name: Name of the model to test
        messages: List of message objects (system, user, etc.)
        temperature: Temperature setting for model inference (default: 0 for deterministic outputs)
        verbose: Whether to print warnings and additional info
        provider: Optional provider override for metrics calculation
        
    Returns:
        Dictionary containing performance metrics:
        - response_content: The model's response text
        - ttft: Time to first token in seconds
        - latency: Total latency in seconds
        - prompt_tokens: Number of input tokens
        - completion_tokens: Number of output tokens
        - throughput: Tokens per second
        - cost: Estimated cost of the API call
        - input_rate: Input token pricing
        - output_rate: Output token pricing
        - provider: Which provider was used
    """
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
            temperature=temperature,
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
        elif verbose:
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
        detected_provider = pricing.get("provider", "hyperbolic")
        
        # Use explicit provider if provided, otherwise use detected provider
        actual_provider = provider if provider else detected_provider
        
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
            "output_rate": output_rate,
            "provider": actual_provider
        }

    except Exception as e:
        if verbose:
            print(f"Error during API call or metric calculation: {e}")
        return {
            "response_content": None,
            "ttft": None,
            "latency": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "throughput": 0,
            "cost": 0,
            "provider": provider or "unknown",
            "error": str(e)
        }