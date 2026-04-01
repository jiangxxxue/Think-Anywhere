# Think Anywhere in Code Generation
[![arXiv](https://img.shields.io/badge/arXiv-2603.29957-b31b1b.svg)](https://arxiv.org/abs/2603.29957)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FF9D00.svg)](https://huggingface.co/papers/2603.29957)
[![alphaXiv](https://img.shields.io/badge/alphaXiv-darkred.svg)](https://www.alphaxiv.org/abs/2603.29957)

This repository contains the source code for the paper **"Think Anywhere in Code Generation"**.
> **Note:** This codebase is being progressively open-sourced. Stay tuned for updates.

> **📢 News:**  Ranked as 🔥 Hot Paper on alphaXiv and 🚀 Trending Paper on Hugging Face!

## Training Your Own Model

We provide two training pipelines depending on whether you choose to use special tokens. Follow the instructions for your preferred configuration, and then proceed to the Reinforcement Learning phase.

### Option 1: With Special Tokens

This pipeline consists of adding special tokens followed by a two-stage supervised fine-tuning (SFT) process.

#### 1. Add Special Tokens
First, add the required special tokens to the base model using the script below:
```bash
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

### Option 2: Without Special Tokens

If you choose not to add special tokens, you can skip the token addition and run a single-stage SFT.

#### 1.Supervised Fine-Tuning (SFT)

Run the standard SFT directly using the configuration file `verl/trainer/config/qwen_code_sft.yaml`:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
  --config-path config \
  --config-name qwen_code_sft.yaml
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

## Citation
If you find this work helpful in your research or use our codebase, we would greatly appreciate a citation!
```bibtex
@article{jiang2026think,
  title={Think Anywhere in Code Generation},
  author={Jiang, Xue and Zhang, Tianyu and Li, Ge and Liu, Mengyang and Chen, Taozhi and Xu, Zhenhua and Li, Binhua and Jiao, Wenpin and Jin, Zhi and Li, Yongbin and Dong, Yihong},
  journal={arXiv preprint arXiv:2603.29957},
  year={2026}
}
```
