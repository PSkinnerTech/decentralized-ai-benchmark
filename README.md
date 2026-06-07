# LLM Inference Benchmark

A CLI tool for benchmarking LLM inference across providers — performance, cost, and accuracy comparisons in a single reproducible run.

Point it at any two models from supported providers and get side-by-side numbers for time-to-first-token, throughput, MMLU accuracy, and per-token cost. Designed for engineers picking an inference provider and researchers comparing model families across infrastructure types.

## What it measures

- **Performance** — time-to-first-token (TTFT), total latency, tokens/sec (streaming)
- **Accuracy** — keyword-based prompt scoring plus full [MMLU](https://github.com/hendrycks/test) evaluation across 57 subjects, with configurable zero-shot or few-shot prompting
- **Cost** — input/output token pricing and cost-performance ratio across providers
- **Reproducibility** — deterministic by default (`temperature=0`, fixed `max_tokens=1024`)

## Supported providers

The tool ships with two reference provider integrations and a pluggable adapter pattern so you can add more:

- **Hyperbolic** — managed API for open-weight models (Llama 3 / 3.1, Mixtral, etc.)
- **Lilypad** — alternative compute network for open-weight models (DeepSeek-R1, Llama 3.1, Qwen 2.5, Mistral, LLaVA, Phi-4-mini)

Adding a new provider is a matter of implementing the request/response adapter — see `providers/` for the existing implementations.

## Quick start

```bash
git clone https://github.com/PSkinnerTech/decentralized-ai-benchmark.git
cd decentralized-ai-benchmark

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install openai python-dotenv requests

cp .env.example .env              # fill in keys for the providers you use
chmod +x hypercompare
```

`.env` only needs the keys for providers you plan to call:

```
HYPERBOLIC_API_KEY=...
LILYPAD_API_KEY=...
```

## Usage

### Compare two models

```bash
# Auto-detect provider per model
./hypercompare meta-llama/Meta-Llama-3.1-8B-Instruct llama3.1:8b

# Explicit provider assignment
./hypercompare meta-llama/Meta-Llama-3.1-8B-Instruct llama3.1:8b \
  --providers hyperbolic lilypad

# Same provider, different models
./hypercompare meta-llama/Meta-Llama-3-70B-Instruct meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Custom prompts

Pipe-delimited keyword scoring for accuracy:

```text
Who wrote Hamlet? | Shakespeare, William Shakespeare
What is 2 + 2? | 4, four
Summarize the benefits of exercise in 2-3 sentences.
```

```bash
./hypercompare model_a model_b --prompts your_prompts.txt
```

### CLI options

```
--providers PROVIDER PROVIDER   Provider for model_a and model_b (hyperbolic | lilypad)
--prompts PATH                  Custom prompt file
--system TEXT                   System prompt applied to all test cases
--temperature FLOAT             Default 0 (deterministic)
--skip-mmlu                     Skip MMLU for a faster run
--n-shots N                     Few-shot examples for MMLU (default 0)
--num-questions N               Questions per MMLU subject (default 5)
--verbose                       Detailed per-request output
```

## MMLU evaluation

Full [MMLU](https://github.com/hendrycks/test) support across all 57 subjects, with:

- Configurable zero-shot or few-shot prompting (`--n-shots`)
- Multi-layered answer extraction (initial-letter, `Answer: X` patterns, regex fallback)
- Per-subject and aggregate accuracy reporting
- Combined cost-vs-accuracy analysis

Example impact of few-shot prompting (5 questions, High School Computer Science):

| Model | Zero-Shot | 3-Shot | Δ |
|---|---|---|---|
| Meta-Llama-3.1-8B | 60.0% | 75.0% | +15.0 |
| Meta-Llama-3-70B | 80.0% | 100.0% | +20.0 |

Run targeted MMLU evaluations directly:

```bash
hypercompare/mmlu_eval.py model_a model_b \
  --subjects high_school_mathematics high_school_physics \
  --num_questions 5 --n_shots 5 --verbose
```

List available subjects:

```bash
python mmlu_dataset.py --list-subjects
```

## Example output

```
============ COMPARISON: Meta-Llama-3.1-8B (hyperbolic) vs llama3.1:8b (lilypad) ============

Speed
  TTFT:        245 ms  vs  312 ms
  Latency:     2.1 s   vs  2.8 s
  Throughput:  95 tps  vs  78 tps

Accuracy
  Prompts:     100.0%  vs  100.0%
  MMLU:         73.2%  vs   71.8%

Cost (per 1K tokens)
  Input:       $0.002  vs  $0.001
  Output:      $0.003  vs  $0.002
  Cost/perf:    1.0x   vs   0.7x
```

## License

MIT © [PSkinnerTech](https://github.com/PSkinnerTech)

## Acknowledgments

Thanks to Hyperbolic and Lilypad for the inference APIs that made the initial reference implementations possible.
