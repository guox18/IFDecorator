<img src="pics/LOGO.png" width="90" style="float: left; margin-right: 10px;" alt="IFDECORATOR Logo">

# IFDECORATOR: Wrapping Instruction Following Reinforcement Learning with Verifiable Rewards


Code for ''IFDECORATOR: Wrapping Instruction Following Reinforcement Learning with Verifiable Rewards''

## 📊 Data

The datasets are available on Hugging Face: [guox18/IFDecorator](https://huggingface.co/datasets/guox18/IFDecorator)

Each data point includes:
- Complex instructions with constraints
- Corresponding Verifications
- Difficulty ratings and metadata

## 🏗️ Code

### Data Processing Pipeline (`modules/`)
- `preprocess/`: Data collection and preprocessing
- `enhance/`: Data augmentation
- `postprocess/`: Post-processing and filtering

### Reinforcement Learning Training (`training/`)
- `reward/`: Reward Design
- `reward_manager/`: A custom reward manager
- Training scripts for Qwen2.5-7B and Qwen2.5-32B models

### Monitoring (`monitoring/`)
- Instructions with trap (`probe.jsonl`)
- Trigger and capture reward hacking.

## 🚀 Quick Start

### Prerequisites

- Python 3.10

### Installation

#### 1. Basic Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd code

# Install dependencies for flywheel
pip install -r requirements.txt
```

### Data Pipeline Execution

The data preparation process consists of three sequential steps:

#### Step 1: Preprocessing
```bash
cd modules/preprocess
./run_preprocess.sh <input_dir> <output_path> [seed]
```

#### Step 2: Enhancement Pipeline
```bash
cd modules/enhance
./run_pipeline.sh
```

#### Step 3: Postprocessing
```bash
cd modules/postprocess
./run_postprocess.sh [pipeline_num] [input_file]
```

### Reinforcement Learning Training
#### 1. Install VERL Environment

```bash
# Clone VERL repository
git clone https://github.com/volcengine/verl.git
cd verl

# Checkout specific commit for compatibility
git checkout 5c5b92819db93dd47ad3403f41ef9b871c47874c

# Install VERL
pip install .
```

#### 2. Configure Reward System

**Important**: Different VERL versions may have different output formats regarding special tokens. Use commit `5c5b92819db93dd47ad3403f41ef9b871c47874c` for guaranteed compatibility

You have two options for reward manager:
- **Option A**: Replace the default reward and reward manager with our custom implementation
- **Option B**: Use the official new batch/manager system (recommended for newer VERL versions)

#### 3. Start Training

Navigate to the recipe directory and run the appropriate training script:

```bash
cd recipe

# For Qwen2.5-7B model
bash run_qwen2_5-7b.sh

# For Qwen2.5-32B model  
bash run_qwen2_5-32b.sh
```

### Reward Hacking Detection

You can monitor and detect potential reward hacking using our tripwires system:

```bash
cd tripwires
./run_hacking_prob.sh
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📚 Citation

If you use this work in your research, please cite:

```bibtex
todo
```
