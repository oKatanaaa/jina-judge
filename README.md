# JinaJudge Training Code

This repository contains the code used to train **JinaJudge**, a model designed to replicate GPT-4-1106-Preview judgments in the Russian LLM Arena for more cost-effective model evaluations.

Human judgments are expensive and time-consuming, and while GPT-4 can act as a proxy, it remains costly. **JinaJudge** reduces costs by replicating these judgment patterns in a lightweight, scalable model.

## Installation

Clone this repository and install the package:

```bash
git clone https://github.com/oKatanaaa/jina-judge
pip install -e .
```

## Usage

To train the model, follow these general steps:

1. **Prepare Data:** Ensure you have datasets formatted as `.jsonl` files (see the `example` for data structure).
2. **Configure Training:** Customize the training parameters in the provided `config.yaml` file.
3. **Run Training:** Use the provided bash script in the `scripts` folder to start training:

```bash
bash run.sh config.yaml
```

The script will save the best-performing model weights, training logs, and outputs in the specified directories. For more details, refer to the example config and dataset structure in the repository.

> NOTE: running the code requires an Ampere GPU or newer due to FlashAttention requirement in the JINA encoder.