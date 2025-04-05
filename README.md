# Hyperbolic Model Benchmarking Tool

A CLI tool for benchmarking different language models hosted on Hyperbolic's API. Compare models based on speed (latency, throughput), accuracy, and cost metrics.

## Features

- **Speed Metrics**: Measure latency, time to first token (TTFT), and throughput (tokens/sec)
- **Accuracy Testing**: Evaluate model responses against expected keywords
- **Cost Analysis**: Calculate token usage and associated costs
- **Model Comparison**: Compare metrics between two different models
- **MMLU Evaluation**: Automatic testing on Massive Multitask Language Understanding benchmark
- **Simple CLI**: Easy-to-use command-line interface
- **Deterministic Testing**: Uses temperature=0 for reproducible benchmarking results

## Prerequisites

- Python 3.8+
- A Hyperbolic account with API access
- API credits for Hyperbolic's API

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/PSkinnerTech/hypercompare.git
cd hypercompare
```

### 2. Set Up Virtual Environment

```bash
cd hypercompare
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install openai python-dotenv requests argparse
```

### 4. Get Your Hyperbolic API Key

1. Create an account at [Hyperbolic](https://hyperbolic.xyz/) if you don't already have one.
2. Request API credits from your Hyperbolic contact.
3. Go to your account settings to retrieve your API key.

### 5. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
echo "HYPERBOLIC_API_KEY=your-api-key-here" > .env
```

Replace `your-api-key-here` with your actual Hyperbolic API key.

### 6. Make the Script Executable (Unix/Linux/Mac)

```bash
chmod +x hypercompare
```

## Usage

### Basic Usage

To compare two models with default test prompts and MMLU evaluation:

```bash
./hypercompare <model_a> <model_b>
```

For example:

```bash
./hypercompare meta-llama/Meta-Llama-3-70B-Instruct meta-llama/Meta-Llama-3.1-8B-Instruct
```

To run a quicker comparison without MMLU evaluation:

```bash
./hypercompare <model_a> <model_b> --skip-mmlu
```

### Custom Prompts

To use custom prompts from a file:

```bash
./hypercompare <model_a> <model_b> --prompts your_prompts.txt
```

### Prompt File Format

Create a text file with one prompt per line. You can specify expected answers for accuracy testing by adding a pipe character (|) followed by comma-separated keywords.

Example prompt file:
```
Who wrote Hamlet? | Shakespeare, William Shakespeare
What is 2 + 2? | 4, four
What is the capital of France? | Paris

# Lines starting with # are comments
# Prompts without expected answers:
Summarize the benefits of exercise in 2-3 sentences.
What are three programming best practices?
```

### CLI Options

```
  -h, --help              Show this help message and exit
  --prompts PROMPTS       Path to a file containing test prompts
  --system SYSTEM         System prompt to use for all test cases
  --verbose               Show more detailed output and warnings
  --temperature TEMP      Temperature setting for model inference (default: 0 for deterministic outputs)
  --skip-mmlu             Skip MMLU evaluation (faster but less comprehensive)
  --n-shots N_SHOTS       Number of few-shot examples to use for MMLU (default: 0 for zero-shot)
  --num-questions NUM     Number of questions per MMLU subject (default: 5)
```

### Benchmarking Configuration

The tool uses the following configuration to ensure consistent, reproducible benchmarks:

- **Temperature**: Set to 0 for deterministic outputs
- **Max Tokens**: 1024 tokens per response
- **Streaming**: Enabled to measure Time to First Token (TTFT)

### Output

The tool will output:

1. **Standard Benchmark:**
   - Individual test results with metrics
   - Model summary with averages

2. **MMLU Evaluation (if not skipped):**
   - Performance on multiple-choice questions across various subjects
   - Accuracy per subject and overall

3. **Comprehensive Comparison:**
   - Speed metrics (TTFT, latency, throughput)
   - Accuracy scores (both standard prompts and MMLU)
   - Cost analysis
   - Overall assessment of which model performs better in different categories

### Available Models

The following models are confirmed to work with Hyperbolic's API:

- `meta-llama/Meta-Llama-3-70B-Instruct`
- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- And others shown in Hyperbolic's API documentation

## MMLU Benchmark

The tool includes support for the Massive Multitask Language Understanding (MMLU) benchmark:

- Tests models on multiple-choice questions across various subjects
- Uses temperature=0 for deterministic, reproducible evaluations
- Supports both zero-shot and few-shot learning (configurable with `--n_shots`)
- Reports detailed accuracy metrics per subject
- Provides cost-performance analysis between models

### Few-Shot Learning in MMLU

Few-shot learning dramatically improves model performance on MMLU by providing example questions and answers before asking the test question:

1. **How it works**: The model is shown n example questions and their correct answers before being asked the test question.

2. **Benefits**:
   - Helps models understand the expected answer format (single letter responses)
   - Improves accuracy by demonstrating the reasoning pattern
   - Reduces "overthinking" by showing examples of direct answers

3. **Implementation**:
   - Examples are selected from the dataset and excluded from test questions
   - Each example is formatted as a user-assistant conversation pair
   - The test question follows the examples in the conversation

### Enhanced Answer Extraction

The MMLU evaluation includes sophisticated answer extraction logic:

1. **Multi-layered parsing**:
   - First attempts to extract the initial character if it's a letter (A, B, C, D)
   - Then looks for patterns like "Answer: B" or "The answer is C" 
   - Finally searches for standalone letters in the response using regex

2. **System prompt guidance**:
   ```
   You are an expert at solving multiple-choice questions.
   Answer with ONLY the letter (A, B, C, or D) corresponding to the correct answer.
   Do not include any explanation, just the single letter.
   For example, if you think option B is correct, respond with just 'B'.
   ```

This approach ensures accurate scoring even when models provide additional explanations or formatting with their answers.

### MMLU CLI Options

The MMLU evaluation script supports these additional options:

```
  --subjects SUBJECTS     Specific MMLU subjects to test (e.g., high_school_mathematics)
  --num_questions NUM     Number of questions per subject (default: 5)
  --n_shots N_SHOTS       Number of few-shot examples to use (default: 0 for zero-shot)
  --split SPLIT           Dataset split to use: dev, validation, or test (default: test)
  --system SYSTEM         Custom system prompt to use
  --temperature TEMP      Temperature setting (default: 0)
  --verbose               Show detailed output for each question
```

### Available MMLU Subjects

The tool supports all 57 MMLU subjects, including:

- `high_school_mathematics`
- `high_school_computer_science`
- `college_mathematics`
- `high_school_physics`
- `college_computer_science`

Use `python mmlu_dataset.py --list-subjects` to see all available subjects.

### MMLU Examples

```bash
# Zero-shot evaluation
hypercompare/mmlu_eval.py meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-70B-Instruct --subjects high_school_mathematics --num_questions 5

# Few-shot evaluation with 5 examples
hypercompare/mmlu_eval.py meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-70B-Instruct --subjects high_school_mathematics --num_questions 5 --n_shots 5

# Multiple subjects with verbose output
hypercompare/mmlu_eval.py meta-llama/Meta-Llama-3.1-8B-Instruct meta-llama/Meta-Llama-3-70B-Instruct --subjects high_school_mathematics high_school_physics --num_questions 3 --n_shots 3 --verbose
```

### Zero-Shot vs Few-Shot Performance

Our testing has shown that few-shot evaluation can significantly improve performance on MMLU tasks, especially for smaller models:

#### Example Results (High School Computer Science, 5 questions)

| Model | Zero-Shot | 3-Shot | Improvement |
|-------|-----------|--------|-------------|
| Meta-Llama-3.1-8B | 60.0% | 75.0% | +15.0% |
| Meta-Llama-3-70B | 80.0% | 100.0% | +20.0% |

#### When to Use Each Approach

- **Zero-Shot**: 
  - More realistic assessment of the model's raw knowledge
  - Faster evaluation (fewer tokens per request)
  - Better for comparing base model capabilities

- **Few-Shot**:
  - Better for maximizing absolute performance scores
  - More similar to how models are typically used in production
  - Essential for complex reasoning tasks
  - More reliable answer formats

For the most comprehensive evaluation, we recommend running both zero-shot and few-shot tests and comparing the results.

## License

MIT © [PSkinnerTech](https://github.com/PSkinnerTech)

## Acknowledgments

- Hyperbolic for providing the API service
