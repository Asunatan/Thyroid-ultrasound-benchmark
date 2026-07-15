# Thyroid Ultrasound Benchmark

A comprehensive benchmark suite for thyroid ultrasound image analysis using large vision-language models. This project provides tools for model training, inference, and evaluation with detailed evaluation metrics.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Workflow Overview](#workflow-overview)

## Installation

### Prerequisites

- Python 3.8+
- CUDA 12.9 compatible GPU(s)
- 50GB+ free disk space for model files

### Step 1: Clone the Repository

```bash
git clone https://github.com/Asunatan/Thyroid-ultrasound-benchmark.git
cd Thyroid-ultrasound-benchmark
```

### Step 2: Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies including:
- PyTorch with CUDA support
- Transformers and related NLP libraries
- vLLM for model serving
- Evaluation metrics (ROUGE, BLEU, BERTScore)
- Data processing tools (torchvision, Pillow, etc.)

## Project Structure

```
Thyroid-ultrasound-benchmark/
├── requirements.txt              # Python dependencies
├── README.md                      # This file
│
├── train_alignment.sh             # Training script for alignment phase
├── train_sft.sh                   # Training script for supervised fine-tuning
│
├── infer.py                       # Inference script for model predictions
├── closed_end_metric.py           # Closed-ended evaluation metrics (Acc, F1)
├── nlp_metrics.py                 # NLP evaluation metrics (ROUGE, BLEU, etc.)
│
├── plugin.py                      # Core plugin utilities
├── ms-swift_debug.py              # MS-Swift debugging utilities
```

## Quick Start

### Complete Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Phase 1: Alignment training
bash train_alignment.sh

# 3. Phase 2: SFT training (after alignment completes)
bash train_sft.sh

# 4. Deploy model with vLLM
CUDA_VISIBLE_DEVICES=4,6 setsid vllm serve /home/user02/SCY/ThyMind-9B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --port 6002 \
  --limit-mm-per-prompt '{"image": 100,"video": 1}' \
  --max-num-seqs 128 \
  --max-model-len 32768 \
  > /home/user02/SCY/thyroid_benchmark/code/swift/serve.log 2>&1

# 5. Wait for vLLM to fully initialize, then run inference
python infer.py \
  --api-port 6002 \
  --input-json data/test_input.json \
  --output-json results/model_output.json

# 6. Evaluate closed-ended metrics
python closed_end_metric.py \
  --pred-json results/model_output.json \
  --gold-json data/ground_truth.json

# 7. Evaluate NLP metrics
python nlp_metrics.py \
  --pred-json results/model_output.json \
  --gold-json data/ground_truth.json
```

## Workflow Overview

### 1. Installation

```bash
pip install -r requirements.txt
```

Installs all dependencies from `requirements.txt`. This is a comprehensive set of packages for:
- Model training (PyTorch, Transformers, DeepSpeed)
- Model inference (vLLM)
- Evaluation (ROUGE, BLEU, BERTScore)
- Image processing (OpenCV, PIL)
- Data handling (Pandas, NumPy)

### 2. Model Training

Training is conducted in two sequential phases: alignment and supervised fine-tuning (SFT).

#### Phase 1: Alignment Training

Run the alignment training script:

```bash
bash train_alignment.sh
```

This phase:
- Aligns the vision encoder with the language model
- Trains the model to understand the relationship between thyroid ultrasound images and text descriptions
- Prepares the foundation for task-specific fine-tuning
- Duration: Varies based on dataset size and hardware

#### Phase 2: Supervised Fine-Tuning (SFT)

After alignment training completes, run the SFT training script:

```bash
bash train_sft.sh
```

This phase:
- Fine-tunes the aligned model on thyroid ultrasound-specific analysis tasks
- Optimizes for accurate medical reporting and diagnosis
- Improves task-specific performance metrics
- Duration: Varies based on dataset size and hardware

### 3. Model Deployment with vLLM

Deploy the trained model using vLLM for high-performance inference:

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

**Command Parameters:**
- `CUDA_VISIBLE_DEVICES=4,6`: Use GPUs 4 and 6 for deployment
- `--tensor-parallel-size 2`: Distribute model across 2 GPUs for faster inference
- `--gpu-memory-utilization 0.9`: Use 90% of GPU memory for maximum efficiency
- `--port 6002`: API server runs on port 6002
- `--limit-mm-per-prompt '{"image": 100,"video": 1}'`: Allow up to 100 images and 1 video per prompt
- `--max-num-seqs 128`: Maximum number of sequences to process in parallel
- `--max-model-len 32768`: Maximum model context length for longer documents
- `> serve.log 2>&1`: Redirect all output to serve.log file and run in background with `setsid`

The service will start in the background. Check status with:
```bash
tail -f /home/user02/SCY/thyroid_benchmark/code/swift/serve.log
```

### 4. Inference

Run inference using the deployed vLLM model:

```bash
python infer.py \
  --api-port 6002 \
  --input-json <path-to-input-json> \
  --output-json <path-to-output-json>
```

**Input Requirements:**
- `--api-port`: Port where vLLM server is running (6002 in example above)
- `--input-json`: Path to JSON file containing thyroid ultrasound images and conversation messages
- `--output-json`: Path where model predictions will be saved

**Input JSON Format:**

The input JSON should follow the message-based conversation format:

```json
[
  {
    "id": "001",
    "image": "<image_url_or_base64>",
    "messages": [
      {
        "role": "user",
        "content": "Describe the thyroid ultrasound findings in this image"
      },
      {
        "role": "assistant",
        "content": "This is a sample assistant response"
      }
    ]
  },
  {
    "id": "002",
    "image": "<image_url_or_base64>",
    "messages": [
      {
        "role": "user",
        "content": "Analyze the nodules visible in this thyroid ultrasound"
      },
      {
        "role": "assistant",
        "content": "Analysis of nodules..."
      }
    ]
  }
]
```

**Key Format Details:**
- `id`: Unique identifier for each sample
- `image`: Image URL or base64-encoded image data
- `messages`: Array of message objects containing the conversation
  - `role`: Either "user" or "assistant"
  - `content`: The message text content

**Output:**

The script generates a JSON file with model predictions:

```json
[
  {
    "id": "001",
    "response": "Model's analysis of the thyroid ultrasound...",
    "timestamp": "2024-01-15T10:30:45"
  },
  {
    "id": "002",
    "response": "Model's analysis of the nodules...",
    "timestamp": "2024-01-15T10:31:12"
  }
]
```

### 5. Evaluation

Evaluate model performance using different evaluation metrics.

#### Phase 1: Closed-ended Metrics

Evaluate closed-ended questions with accuracy and F1 scores:

```bash
python closed_end_metric.py \
  --pred-json <path-to-model-output-json> \
  --gold-json <path-to-ground-truth-json>
```

**Output Metrics:**
- Accuracy (Acc): Percentage of correct predictions
- F1 Score: Harmonic mean of precision and recall
- Precision: True positives / All predicted positives
- Recall: True positives / All actual positives
- Confusion Matrix: Breakdown of predictions vs actual values
- Per-class metrics: Individual performance for each classification class

**Example Output:**
```json
{
  "accuracy": 0.92,
  "f1_macro": 0.89,
  "f1_weighted": 0.91,
  "precision_macro": 0.90,
  "recall_macro": 0.88,
  "per_class_metrics": {...}
}
```

#### Phase 2: NLP Metrics

Evaluate open-ended responses with NLP metrics:

```bash
python nlp_metrics.py \
  --pred-json <path-to-model-output-json> \
  --gold-json <path-to-ground-truth-json>
```

**Output Metrics:**
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
  - ROUGE-1: Unigram overlap
  - ROUGE-2: Bigram overlap
  - ROUGE-L: Longest common subsequence
- BLEU (Bilingual Evaluation Understudy): Precision-based n-gram overlap
- BERTScore: Contextual embedding-based similarity
- Semantic Similarity: Model-based semantic alignment
- Additional metrics for text quality assessment

**Example Output:**
```json
{
  "rouge": {
    "rouge1": 0.56,
    "rouge2": 0.38,
    "rougeL": 0.52
  },
  "bleu": 0.42,
  "bertscore": {
    "precision": 0.88,
    "recall": 0.87,
    "f1": 0.88
  },
  "semantic_similarity": 0.82
}
```

## Configuration and Customization

### Key Configuration Scripts

- **`train_alignment.sh`**: Configures alignment training parameters, data paths, and learning rates
- **`train_sft.sh`**: Configures SFT training parameters, task definitions, and optimization settings
- **`infer.py`**: Handles vLLM API communication, batch processing, and output formatting
- **`closed_end_metric.py`**: Computes closed-ended evaluation metrics with statistical analysis
- **`nlp_metrics.py`**: Computes open-ended NLP evaluation metrics and text quality scores
- **`plugin.py`**: Core utility functions and helper modules
- **`ms-swift_debug.py`**: Debugging utilities for MS-Swift framework integration

### Customizing Inference

Edit inference parameters in `infer.py`:
- Temperature: Controls randomness of responses
- Top-p (nucleus sampling): Adjusts diversity of outputs
- Max tokens: Maximum length of generated responses
- Batch size: Number of samples processed simultaneously

## Troubleshooting

### GPU Memory Issues

If you encounter out-of-memory errors during deployment:

```bash
# Solution 1: Reduce batch size
--max-num-seqs 64

# Solution 2: Reduce GPU memory utilization
--gpu-memory-utilization 0.7

# Solution 3: Reduce context length
--max-model-len 16384
```

### vLLM Service Won't Start

**Diagnosis:**
```bash
# Check GPU availability
nvidia-smi

# Check if port is already in use
lsof -i :6002
```

**Solutions:**
- Verify CUDA_VISIBLE_DEVICES are correct and GPUs are available
- Check available GPU memory (recommend 40GB+ per GPU for 9B models)
- Review `serve.log` for detailed error messages: `tail -f serve.log`
- Try with fewer GPUs or reduced memory utilization

### Inference Timeout

- Verify vLLM service is running: `curl http://localhost:6002/v1/models`
- Check network connectivity if running remotely
- Increase `--max-model-len` if handling longer sequences
- Reduce batch size in `infer.py`

### Low Evaluation Scores

- Verify training completed successfully
- Check input data format matches training expectations
- Ensure ground truth JSON has correct structure and labels
- Review model outputs for potential format mismatches

## Performance Tips

1. **GPU Utilization**: Adjust `--gpu-memory-utilization` (0.7-0.95) based on your setup and stability needs
2. **Batch Processing**: Process multiple samples in single vLLM request when possible for better throughput
3. **Model Caching**: vLLM automatically caches compiled kernels; first run may be slower
4. **Tensor Parallelism**: Use `--tensor-parallel-size` equal to number of available GPUs for optimal performance
5. **Context Length**: Set `--max-model-len` based on your typical input length to save memory

## Dependencies

Key packages included in `requirements.txt`:
- **vLLM 0.21.0**: High-throughput LLM serving
- **PyTorch 2.11.0**: Deep learning framework
- **Transformers 5.8.1**: NLP models and utilities
- **MS-Swift 4.4.1**: Fine-tuning framework
- **ROUGE/BLEU/BERTScore**: Evaluation metrics
- **OpenCV & PIL**: Image processing
- **NLTK & Jieba**: Text processing

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

For questions or issues, please open an issue on GitHub or contact the maintainers.
