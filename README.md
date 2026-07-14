# Thyroid Ultrasound Benchmark

A comprehensive benchmark suite for thyroid ultrasound image analysis using advanced deep learning models with Vision Language Models (VLMs).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Model Serving](#model-serving)
- [Output](#output)

## Overview

This repository provides a complete pipeline for:
- Training thyroid ultrasound analysis models using alignment and supervised fine-tuning (SFT)
- Serving pre-trained Vision Language Models with vLLM
- Running inference on thyroid ultrasound images
- Evaluating model performance with comprehensive metrics

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0 or higher (for GPU support)
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Asunatan/Thyroid-ultrasound-benchmark.git
cd Thyroid-ultrasound-benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

This will install all required packages for training, inference, and evaluation.

## Project Structure

```
Thyroid-ultrasound-benchmark/
├── README.md
├── requirements.txt
├── code/
│   ├── swift/
│   │   ├── align.sh          # Alignment training script
│   │   ├── sft.sh            # Supervised fine-tuning script
│   │   ├── infer.py          # Inference script
│   │   ├── serve.log         # vLLM server logs
│   │   └── ...
│   └── ...
├── data/
│   └── # Your dataset files
└── results/
    └── # Output results and metrics
```

## Usage

### Training

The training process consists of two stages:

#### Stage 1: Alignment

Run the alignment training script to align image and text representations:

```bash
bash code/swift/align.sh
```

This stage trains the model to understand the relationship between thyroid ultrasound images and their corresponding text descriptions.

#### Stage 2: Supervised Fine-Tuning (SFT)

After alignment completes, run the supervised fine-tuning script:

```bash
bash code/swift/sft.sh
```

This stage fine-tunes the model on the specific thyroid ultrasound analysis tasks.

### Model Serving

Deploy the trained model using vLLM for high-performance inference:

#### Start the vLLM Server

```bash
CUDA_VISIBLE_DEVICES=4,6 setsid vllm serve /home/user02/SCY/ThyMind-9B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --port 6002 \
  --limit-mm-per-prompt '{"image": 100,"video": 1}' \
  --max-num-seqs 128 \
  --max-model-len 32768 \
  > /home/user02/SCY/thyroid_benchmark/code/swift/serve.log 2>&1
```

**Parameters explained:**
- `CUDA_VISIBLE_DEVICES=4,6`: Use GPUs 4 and 6
- `--tensor-parallel-size 2`: Distribute across 2 GPUs
- `--gpu-memory-utilization 0.9`: Use 90% of GPU memory
- `--port 6002`: Server runs on port 6002
- `--limit-mm-per-prompt`: Set multimedia input limits (100 images, 1 video per prompt)
- `--max-num-seqs 128`: Maximum number of sequences per batch
- `--max-model-len 32768`: Maximum context length

The server will run in the background and log to `serve.log`.

### Inference

Run inference on thyroid ultrasound images using the deployed model:

```bash
python code/swift/infer.py \
  --input-json <path-to-input-json> \
  --port 6002 \
  --output-json <path-to-output-json>
```

**Arguments:**
- `--input-json`: Path to the input JSON file containing image paths and metadata
- `--port`: Port number where vLLM server is running (default: 6002)
- `--output-json`: Path where the model outputs will be saved

**Example input JSON format:**
```json
[
  {
    "id": "001",
    "image_path": "/path/to/image1.jpg",
    "prompt": "Describe the thyroid ultrasound findings"
  },
  {
    "id": "002",
    "image_path": "/path/to/image2.jpg",
    "prompt": "Analyze this thyroid ultrasound image"
  }
]
```

The model will generate predictions and save them to the output JSON file.

### Evaluation

Evaluate model performance using multiple metrics:

#### Closed-set Evaluation

Calculate accuracy, F1 score, precision, recall, and other classification metrics:

```bash
python code/swift/eval_closed.py \
  --predictions <path-to-model-output-json> \
  --ground-truth <path-to-ground-truth-json> \
  --output-metrics <path-to-save-metrics>
```

This produces metrics including:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

#### NLP Evaluation

Evaluate the quality of generated text descriptions using NLP metrics:

```bash
python code/swift/eval_nlp.py \
  --predictions <path-to-model-output-json> \
  --ground-truth <path-to-ground-truth-json> \
  --output-metrics <path-to-save-nlp-metrics>
```

This produces NLP-specific metrics including:
- BLEU Score
- ROUGE Score
- METEOR Score
- BERTScore
- Semantic Similarity

## Output

The evaluation scripts produce JSON files with detailed metrics:

**Closed-set Metrics (eval_closed.py output):**
```json
{
  "accuracy": 0.95,
  "precision": {
    "macro": 0.93,
    "weighted": 0.94
  },
  "recall": {
    "macro": 0.92,
    "weighted": 0.95
  },
  "f1": {
    "macro": 0.92,
    "weighted": 0.94
  },
  "confusion_matrix": [...],
  "per_class_metrics": {...}
}
```

**NLP Metrics (eval_nlp.py output):**
```json
{
  "bleu": 0.42,
  "rouge": {
    "rouge1": 0.56,
    "rouge2": 0.38,
    "rougeL": 0.52
  },
  "meteor": 0.45,
  "bertscore": {
    "precision": 0.88,
    "recall": 0.87,
    "f1": 0.88
  },
  "semantic_similarity": 0.82
}
```

## Common Issues

### GPU Memory Issues

If you encounter out-of-memory errors:
- Reduce `--max-num-seqs` parameter
- Decrease `--gpu-memory-utilization` value
- Reduce `--max-model-len` parameter

### Port Already in Use

If port 6002 is already in use:
```bash
# Find and kill the process using the port
lsof -ti:6002 | xargs kill -9
```

Then restart the vLLM server with a different port.

### Server Connection Errors

Ensure the vLLM server is running and accessible:
```bash
# Check if server is running
curl http://localhost:6002/v1/models
```

## Requirements

See `requirements.txt` for the complete list of dependencies. Key packages include:
- vLLM
- PyTorch
- Transformers
- NumPy
- scikit-learn
- nltk
- rouge-score

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@repository{thyroid_ultrasound_benchmark,
  title={Thyroid Ultrasound Benchmark},
  author={Asunatan},
  year={2024},
  url={https://github.com/Asunatan/Thyroid-ultrasound-benchmark}
}
```

## License

Please check the LICENSE file for licensing information.

## Contact

For questions and feedback, please open an issue on the GitHub repository.
