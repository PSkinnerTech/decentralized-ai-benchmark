# Hyperbolic Model Benchmarking Tool

A command-line interface (CLI) tool for benchmarking different language models hosted on Hyperbolic's API. Compare models based on speed (latency, throughput), accuracy, and cost metrics.

## Features

- **Speed Metrics**: Measure latency, time to first token (TTFT), and throughput (tokens/sec)
- **Accuracy Testing**: Evaluate model responses against expected keywords
- **Cost Analysis**: Calculate token usage and associated costs
- **Model Comparison**: Compare metrics between two different models
- **Simple CLI**: Easy-to-use command-line interface

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

## Usage

TBD

## License

MIT © [PSkinnerTech](https://github.com/PSkinnerTech)

## Acknowledgments

- Hyperbolic for providing the API service
- OpenAI for the Python library used to interact with the API
