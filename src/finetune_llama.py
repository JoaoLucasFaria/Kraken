import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
import pandas as pd
import os
import gc

def preparar_dados_treino_llama31(csv_path="monster_descriptions.csv", output_dir="outputs"):
    print("[INFO] Carregando dados de descrições para Llama 3.1...")
    
    df = pd.read_csv(csv_path)
    
    training_data = []
    for _, row in df.iterrows():
        monster_name = row.get("monster_name", "Unknown")
        description = row.get("description", "No description available")
        
        prompt = f"""<|start_header_id|>user<|end_header_id|>

Describe the D&D monster: {monster_name}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{description}<|eot_id|>"""
        
        training_data.append({"text": prompt})
    
    output_path = Path(output_dir) / "train_data_llama31.jsonl"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"[OK] Dados salvos em {output_path}")
    return str(output_path)

def finetune_llama31(train_data_path: str, output_dir: str = "outputs/llama31_finetuned", num_epochs: int = 1):
    """Fine-tune Llama 3.1 - VERSÃO ULTRA OTIMIZADA PARA Q3_K_M"""
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("[INFO] Iniciando fine-tuning com Llama 3.1...")
    print("[AVISO] Fine-tuning com 7.6GB VRAM é EXTREMAMENTE desafiador.")
    print("[RECOMENDAÇÃO] Use o modelo GGUF pré-treinado em model_loader.py!")
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    print(f"[INFO] Carregando {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 128  # REDUZIDO de 200 para 128
    
    # Quantização 4-bit MAIS AGRESSIVA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,  # Desabilitado para economizar VRAM
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory={0: "6GB"},  # LIMITE DE 6GB PARA DEIXAR MARGEM
        )
    except Exception as e:
        print(f"[ERRO] Falha ao carregar modelo: {e}")
        print("[SOLUÇÃO] Use o modelo GGUF Q3_K_M em model_loader.py!")
        return None
    
    # Preparar modelo para treinamento 4-bit SEM gradient checkpointing
    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    except torch.cuda.OutOfMemoryError:
        print("[ERRO] CUDA OOM durante prepare_model_for_kbit_training")
        print("[SOLUÇÃO] GPU de 7.6GB é insuficiente para fine-tuning de Llama 3.1 8B")
        return None
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # LoRA ULTRA MÍNIMO
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,  # Mínimo absoluto
        lora_alpha=4,
        lora_dropout=0.05,
        target_modules=["q_proj"],  # Apenas q_proj (mínimo)
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Verificar parâmetros treináveis
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parâmetros treináveis: {trainable_params:,}")
    
    if trainable_params == 0:
        print("[ERRO] Nenhum parâmetro treinável!")
        return None
    
    model.print_trainable_parameters()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("[INFO] Carregando dados...")
    data = []
    with open(train_data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    # APENAS 3 EXEMPLOS (teste mínimo)
    if len(data) > 3:
        print(f"[WARN] Limitando dataset de {len(data)} para 3 exemplos (teste)")
        data = data[:3]
    
    dataset = Dataset.from_list(data)
    
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,  # REDUZIDO
            padding="max_length"
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # TrainingArguments ULTRA MÍNIMO
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # Sem acumulação
        learning_rate=5e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="no",
        optim="paged_adamw_8bit",
        warmup_steps=0,
        max_grad_norm=0.3,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        max_steps=3,  # APENAS 3 STEPS (teste)
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    print("[INFO] Iniciando treinamento (TESTE COM 3 EXEMPLOS)...")
    print(f"[INFO] VRAM em uso: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    try:
        trainer.train()
        print(f"[INFO] Salvando adaptador LoRA...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[OK] Fine-tuning concluído!")
        return output_dir
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[ERRO] CUDA OOM durante treinamento: {e}")
        return None
    except Exception as e:
        print(f"\n[ERRO] {str(e)}")
        return None

if __name__ == "__main__":
    train_path = preparar_dados_treino_llama31(
        csv_path="monster_descriptions.csv",
        output_dir="outputs"
    )
    
    finetuned_model_path = finetune_llama31(
        train_data_path=train_path,
        num_epochs=1
    )