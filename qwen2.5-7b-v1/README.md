# Qwen/Qwen2.5-7B - Finance (Run v1)

This directory contains the federated instruction tuning submission for the **Finance** challenge using the [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) model on the [flwrlabs/fingpt-sentiment-train](https://huggingface.co/datasets/flwrlabs/fingpt-sentiment-train) dataset.

We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in a federated way,
allowing users to perform the training on a single GPU.

## Project Structure

```text
.
├── mmfl/                            # Source code for ClientApp, ServerApp, and Strategy
├── flowertune-eval-finance/          # Evaluation scripts and instructions
├── pyproject.toml                    # Project configuration and dependencies
└── README.md                         # This file
```

## Methodology

This submission performs federated LLM fine-tuning with **LoRA** using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.
The clients' models are aggregated with the **FedAvg** strategy.

### Model Configuration
- **Base Model**: `Qwen/Qwen2.5-7B`
- **Quantization**: 4-bit
- **PEFT**: LoRA (Rank: 32, Alpha: 64)
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Training Configuration
- **Rounds**: 10
- **Fraction Fit**: 0.1 (10% of clients per round)
- **Local Epochs**: 3
- **Optimizer**: Paged AdamW 8-bit

## Prerequisites

Before running the simulation, ensure you have access to the model and are logged into Hugging Face.

1. **Model Access**: Ensure you have access to [InternLM3 8B Instruct](https://huggingface.co/internlm/internlm3-8b-instruct) on Hugging Face.
2. **Hugging Face Login**:
   ```bash
   huggingface-cli login
   ```

## Setup & Running

1. **Install Dependencies**:
   Ensure you are in this directory (`submissions/finance/qwen2.5-7b-v1`).
   ```bash
   pip install -e .
   ```

2. **Run Simulation**:
   Run the challenge with default config values defined in `pyproject.toml`.
   ```bash
   flwr run
   ```

> [!IMPORTANT]
> Please note that `[tool.flwr.app.config.static]` and `options.num-supernodes` under `[tool.flwr.federations.local-simulation]` in `pyproject.toml` are not allowed to be modified for fair competition if you plan to participate in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).

## Experimental Setup

The dataset is divided into 20 partitions in an IID fashion, a partition is assigned to each ClientApp.
We randomly sample a fraction (0.1) of the total nodes to participate in each round, for a total of 10 rounds.

## VRAM Consumption & Resources

You can adjust the CPU/GPU resources assigned to each client based on your device capabilities by modifying `options.backend.client-resources.num-cpus` and `options.backend.client-resources.num-gpus` under `[tool.flwr.federations.local-simulation]` entry in `pyproject.toml`.

Experiments were run on RTX 3090/4090 class GPUs with 4-bit quantization.

## Model Saving

The global PEFT model checkpoints are saved every 5 rounds after aggregation on the server side as default, which can be specified with `train.save-every-round` under `[tool.flwr.app.config]` entry in `pyproject.toml`.

> [!NOTE]
> Please provide the last PEFT checkpoint if you plan to participate in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).

## Changes from Baseline
- Base model: switched from `mistralai/Mistral-7B-v0.3` to `Qwen/Qwen2.5-7B`.
- Rounds: reduced from 200 to 10.
- LoRA: rank/alpha `32/64` and target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (baseline: `32/64`, default targets).
- Learning Rate: increased compared to the baseline, from `5e-5 / 1e-6` to `5e-4 / 5e-5` (max / min).
- Torch/runtime stack: `torch==2.4.0`, `peft==0.14.0`, `transformers==4.50.3` (baseline uses `torch==2.9.1`, `peft==0.6.2`).

## Evaluation

See `evaluation/README.md` for the exact environment setup and the single-line command to run. Results are stored under `evaluation/benchmarks/` (acc/generation artifacts already included).

### Results (peft_10)

|          | fiqa  |  fpb  | tfns  | Average |
|:--------:|:-----:|:-----:|:-----:|:-------:|
|  FedAvg  | 83.22 | 85.89 | 84.63 |  84.58  |

**Communication budget: 30807.07 MB**

## Checkpoints

- Round 10 PEFT adapter: [Google Drive link](https://drive.google.com/file/d/115J-Goo8KwIDx5BvPqaM9xT34QoaisNy/view?usp=sharing)