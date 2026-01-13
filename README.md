# Think Anywhere in Code Generation

This repository contains the source code for the paper **"Think Anywhere in Code Generation"**.

## Training Your Own Model

If you want to train your own model, follow these steps:

### 1. Download Training Data
```python
thinkanywhere_scripts/data_preprocess/download_and_filter_data_7b.py
```

### 2. Run Training Script
Execute the training script with the following command:
```bash
bash thinkanywhere_scripts/train/run.sh
```
**Note:** Make sure to modify all file paths in the script with your specific paths and configurations.

