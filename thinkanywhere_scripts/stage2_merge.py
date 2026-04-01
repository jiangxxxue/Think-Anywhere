import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_stage2(base_model_path, stage2_ckpt_path, output_path):
    print(f"🔄 开始 Stage 2 合并...")
    print(f"📂 基础模型: {base_model_path}")
    print(f"📂 PEFT 权重: {stage2_ckpt_path}")

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(stage2_ckpt_path)

    # 2. 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu",
        trust_remote_code=True
    )

    if len(tokenizer) > base_model.config.vocab_size:
        print(f"⚠️ 检测到词表大小改变 (Model: {base_model.config.vocab_size} -> Tokenizer: {len(tokenizer)})")
        base_model.resize_token_embeddings(len(tokenizer))
        print("✅ 已重新调整模型的 Embedding 尺寸")

    # ==========================================
    # 🔧 [新增逻辑] 自动修复 adapter_model.bin 的 Key 后缀
    # ==========================================
    adapter_bin_path = os.path.join(stage2_ckpt_path, "adapter_model.bin")
    if os.path.exists(adapter_bin_path):
        state_dict = torch.load(adapter_bin_path, map_location="cpu")
        # 检查是否包含未被裁剪的后缀
        if any(".modules_to_save.default" in k for k in state_dict.keys()):
            print("🔧 检测到由于手动 FSDP 序列化残留的 PEFT 后缀，正在自动修复 Key ...")
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace(".modules_to_save.default", "")
                new_state_dict[new_k] = v
            # 覆盖保存
            torch.save(new_state_dict, adapter_bin_path)
            print("✅ Key 后缀修复完成！")
    # ==========================================

    # 3. 将模型包装为 PEFT 模型
    print("⏳ 正在加载 PEFT Adapter 及附加模块权重...")
    peft_model = PeftModel.from_pretrained(
        base_model, 
        stage2_ckpt_path, 
        device_map="cpu"
    )

    # 4. 执行物理合并
    print("⚙️ 正在执行 merge_and_unload() 矩阵融合...")
    merged_model = peft_model.merge_and_unload()
    print("✅ 矩阵融合完成！")

    # 5. 保存最终模型
    print(f"💾 正在保存最终大模型至: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print("🎉 Stage 2 合并完成！可以用作最终推理或 vLLM 部署了！\n")

# 使用示例：
merge_stage2(
    base_model_path="path_to_stage1_merged",
    stage2_ckpt_path="path_to_stage2_ckpt",
    output_path="path_to_output"
)