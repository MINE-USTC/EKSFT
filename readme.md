
# EK-SFT 


### Implementation Note

This project is built upon and extends the LlamaFactory framework. The codebase follows LlamaFactory's architecture and design patterns, with EKSFT-specific modifications implemented in the `src/llamafactory/train/eksft/` directory.




## 🚀 Quick Start
### Environment Configuration

#### 1. Python Dependencies Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows activation

# Install core dependencies
cd EK-SFT
pip install -e .
pip install -r requirements/metrics.txt
```

#### 2. Project-Specific Dependencies

```bash
# Install development dependencies
pip install -r requirements/dev.txt

# Install training dependencies
pip install -r requirements/deepspeed.txt  # If using DeepSpeed
```

### Data Preparation

The implementation utilizes the `openr1_math_3k_sft` dataset, with data files located at:
- `data/sup_3k_data.json` - Primary training data

Dataset configurations are defined in `data/dataset_info.json`.

## 🏃 Training Execution

### 1. Multi-GPU Training (with DeepSpeed)

```bash
bash ./examples/example_eksft/eksft.sh
```

## ⚙️ Configuration Parameters

### EKSFT-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eksft_top_k_ratio` | 0.2 | Proportion of tokens selected based on KL divergence and entropy metrics |
| `--eksft_lambda_entropy` | 0.05 | Weight coefficient for entropy regularization loss |
| `--eksft_lambda_kl` | 0.05 | Weight coefficient for KL divergence regularization loss |
| `--eksft_is_union_mask` | true | Use union (true) or intersection (false) of KL and entropy masks |
| `--eksft_largest_kl` | true | Select tokens with largest (true) or smallest (false) KL divergence |
| `--eksft_largest_entropy` | true | Select tokens with largest (true) or smallest (false) entropy |
| `--eksft_output_dir` | None | Output directory for saving KL and entropy logs during training |

### General Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name_or_path` | - | Pretrained model path or HuggingFace model identifier |
| `--stage` | eksft | Training stage, must be set to `eksft` |
| `--finetuning_type` | full | Fine-tuning type: `full` (full-parameter) or `lora` |
| `--dataset` | openr1_math_3k_sft | Dataset name |
| `--template` | qwen2.5 | Model template |
| `--cutoff_len` | 20000 | Maximum sequence length |
| `--per_device_train_batch_size` | 1 | Training batch size per device |
| `--gradient_accumulation_steps` | 8 | Gradient accumulation steps |
| `--learning_rate` | 1e-5 | Learning rate |
| `--num_train_epochs` | 8.0 | Number of training epochs |

## 🗂️ Project Structure

```
EKSFT/
├── src/                    # Source Code
│   ├── train.py           # Training Entry Point
│   ├── api.py             # API Interface
│   └── llamafactory/      # LlamaFactory Core Implementation
│       ├── train/eksft/   # EKSFT Training Implementation
│       │   ├── trainer.py # EKSFT Trainer
│       │   └── workflow.py# Training Workflow
│       └── hparams/       # Hyperparameter Definitions
├── data/                  # Data Files
│   ├── dataset_info.json # Dataset Configuration
│   └── sup_3k_data.json  # Training Data
├── examples/              # Example Configurations
│   ├── example_eksft/    # EKSFT Examples
│   │   └── eksft.sh      # Linux Training Script
│   ├── accelerate/       # Accelerate Configurations
│   └── deepspeed/        # DeepSpeed Configurations
├── requirements/          # Dependency Files
│   ├── dev.txt           # Development Dependencies
│   ├── deepspeed.txt     # DeepSpeed Dependencies
│   └── ...               # Other Optimizer Dependencies
├── scripts/              # Utility Scripts
│   ├── eval_bleu_rouge.py# Evaluation Scripts
│   └── stat_utils/       # Statistical Utilities
└── tests/                # Test Files
```

## 🔧 Advanced Usage

### Custom Dataset Integration

1. Prepare data file (JSON format):
```json
[
  {
    "conversations": [
      {"role": "user", "value": "Question content"},
      {"role": "assistant", "value": "Answer content"}
    ]
  }
]
```

2. Add configuration to `data/dataset_info.json`:
```json
"your_dataset_name": {
  "file_name": "your_data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations"
  }
}
```

### Model Architecture Support

The framework supports multiple model architectures:
- Qwen Series: `--template qwen2.5` or `--template qwen3_nothink`
- Llama Series: `--template llama3`
- ChatGLM Series: `--template chatglm3`

### Selection Strategy Adjustment

1. **Intersection Mode** (more stringent selection):
```bash
--eksft_is_union_mask false
```

2. **Select Low-Entropy/Low-KL Tokens**:
```bash
--eksft_largest_kl false --eksft_largest_entropy false
```
