# Think Anywhere in Code Generation

This repository contains the source code for the paper **"Think Anywhere in Code Generation"**.
> **Note:** This codebase is being progressively open-sourced. Stay tuned for updates.

## Training Your Own Model

The training pipeline consists of three sequential stages: adding special tokens, supervised fine-tuning (SFT), and reinforcement learning (RL). Follow these steps to train your own model:

### 1. Add Special Tokens
First, add the required special tokens to the base model using the script below:
```python
python thinkanywhere_scripts/add_special_token.py
```

### 2. Supervised Fine-Tuning (SFT)
We conduct SFT in two stages. Set up your environment and execute the training commands as follows:
#### 2.1 Stage 1 SFT
Use the configuration file `verl/trainer/config/qwen_code_sft_stage1.yaml` and run:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
  --config-path config \
  --config-name qwen_code_sft_stage1.yaml
```

#### 2.2 Stage 2 SFT
After completing Stage 1, proceed to Stage 2 with the configuration file `verl/trainer/config/qwen_code_sft_stage2.yaml`:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
  --config-path config \
  --config-name qwen_code_sft_stage2.yaml
```
### 3. Reinforcement Learning (RL) Training
Once SFT is finished, start the RL training phase:

#### 3.1 Download Training Data
```python
python thinkanywhere_scripts/data_preprocess/download_and_filter_data_7b.py
```

#### 3.2 Run RL Training Script
Execute the RL training script with the following command:
```bash
bash thinkanywhere_scripts/train/run.sh
```

**Note:** Make sure to modify all file paths in the script with your specific paths and configurations.

