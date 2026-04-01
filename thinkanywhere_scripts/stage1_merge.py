import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

def merge_stage1(base_model_path, stage1_ckpt_path, output_path):
    print(f"🔄 开始 Stage 1 合并...")
    print(f"📂 基础模型: {base_model_path}")
    print(f"📂 局部权重: {stage1_ckpt_path}")

    # 1. 加载 Tokenizer
    # 使用 ckpt 的 tokenizer，因为它可能包含了新增的特殊 Token (如 <thinkanywhere>)
    tokenizer = AutoTokenizer.from_pretrained(stage1_ckpt_path)

    # 2. 加载基础模型 (建议用 bfloat16 加载到 CPU 内存以节省资源)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu",
        trust_remote_code=True
    )

    # ⚠️ 关键检查：如果新增了 Token，必须先扩充模型的 Embedding 矩阵形状
    if len(tokenizer) > model.config.vocab_size:
        print(f"⚠️ 检测到词表大小改变 (Model: {model.config.vocab_size} -> Tokenizer: {len(tokenizer)})")
        model.resize_token_embeddings(len(tokenizer))
        print("✅ 已重新调整模型的 Embedding 尺寸")

    # 3. 加载我们 Stage 1 保存的局部权重 (State Dict)
    safetensors_path = os.path.join(stage1_ckpt_path, "model.safetensors")
    bin_path = os.path.join(stage1_ckpt_path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        partial_state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        partial_state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError("在 Checkpoint 目录中找不到权重文件！")

    # 4. 将局部权重覆盖回基础模型 (strict=False 是必须的，因为我们只加载了部分层)
    missing_keys, unexpected_keys = model.load_state_dict(partial_state_dict, strict=False)
    
    print(f"✅ 已成功覆盖 {len(partial_state_dict)} 个 Tensor。")
    if unexpected_keys:
        print(f"⚠️ 警告: 发现了无法匹配的 Key: {unexpected_keys}")

    # 5. 保存完整合并后的模型
    print(f"💾 正在保存完整模型至: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True) # 默认保存为 safetensors
    tokenizer.save_pretrained(output_path)
    print("🎉 Stage 1 合并完成！\n")

merge_stage1(
    base_model_path="path_to_base_model",
    stage1_ckpt_path="path_to_ckpt",
    output_path="path_to_output"
)