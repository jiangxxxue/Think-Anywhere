import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AddedToken


BASE_MODEL_PATH = "path_to_base_model"
SAVE_PATH = "path_to_save_dir"

NEW_TOKEN_NAMES = ["<thinkanywhere>", "</thinkanywhere>"]

SEMANTIC_SOURCE = "thinkanywhere"

FUNCTIONAL_MAPPING = {
    "<thinkanywhere>": "<|im_start|>",
    "</thinkanywhere>": "<|im_end|>"
}

def main():
    print(f"🚀 Loading base model: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, 
        trust_remote_code=True, 
        use_fast=True, 
        padding_side="right"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"📊 原始词表大小: {len(tokenizer)}")

    print("\n➕ Adding tokens with Explicit Special Status...")
    
    new_token_objects = []
    for t_name in NEW_TOKEN_NAMES:
        token_obj = AddedToken(t_name, lstrip=False, rstrip=False, special=True, normalized=False)
        new_token_objects.append(token_obj)
    
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_token_objects})
    
    print(f"✅ 成功注册了 {len(new_token_objects)} 个 Special Tokens")
    print(f"   (实际新增词表 ID 数: {num_added})")

    model.resize_token_embeddings(len(tokenizer))
    print(f"📏 Resized embeddings to: {len(tokenizer)}")

    print(f"\n🧠 Calculating semantic embedding for '{SEMANTIC_SOURCE}'...")
    
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    
    semantic_ids = tokenizer.encode(SEMANTIC_SOURCE, add_special_tokens=False)
    semantic_tokens = tokenizer.convert_ids_to_tokens(semantic_ids)
    print(f"   Breakdown: {SEMANTIC_SOURCE} -> {semantic_tokens} (IDs: {semantic_ids})")
    
    input_semantic_vecs = torch.stack([input_embeddings[idx] for idx in semantic_ids])
    input_semantic_mean = torch.mean(input_semantic_vecs, dim=0)
    
    output_semantic_vecs = torch.stack([output_embeddings[idx] for idx in semantic_ids])
    output_semantic_mean = torch.mean(output_semantic_vecs, dim=0)
    
    print("   ✅ Semantic mean calculated.")

    print("\n⚗️  Mixing Embeddings (Semantic + Functional) / 2 ...")
    
    for new_tok, func_tok in FUNCTIONAL_MAPPING.items():
        new_id = tokenizer.convert_tokens_to_ids(new_tok)
        func_id = tokenizer.convert_tokens_to_ids(func_tok)
        
        if new_id == tokenizer.unk_token_id:
            print(f"❌ Error: {new_tok} ID retrieval failed!")
            continue

        input_func_vec = input_embeddings[func_id]
        output_func_vec = output_embeddings[func_id]
        
        new_input_vec = (input_semantic_mean + input_func_vec) / 2.0
        new_output_vec = (output_semantic_mean + output_func_vec) / 2.0
        
        input_embeddings[new_id] = new_input_vec
        output_embeddings[new_id] = new_output_vec
        
        print(f"   ✨ {new_tok} (ID:{new_id}) initialized from: avg('{SEMANTIC_SOURCE}') + '{func_tok}'")

    print("\n🔍 Pre-Save Verification:")
    for t in NEW_TOKEN_NAMES:
        is_special = t in tokenizer.all_special_tokens
        if is_special:
            print(f"   ✅ {t} IS marked as Special.")
        else:
            print(f"   ❌ {t} is NOT marked as Special! Something is wrong.")
            
    print(f"\n💾 Saving to: {SAVE_PATH}")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("🎉 Done! This is your Golden Base Model (Special Token Guaranteed).")

if __name__ == "__main__":
    main()