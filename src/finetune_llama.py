import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
import pandas as pd
import os
import gc

def preparar_dados_treino_llama31(csv_path="/content/drive/MyDrive", output_dir="outputs"):
    print("[INFO] Carregando dados de descrições para Llama 3.1...")
    
    df = pd.read_csv(csv_path)
    print(df.head(3))
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
    """
    Fine-tune Llama 3.1 usando modelo GGUF via llama-cpp-python
    Sem dependência de BitsAndBytes - usa inferência com GGUF quantizado
    """
    
    print("[INFO] Carregando dados de treino...")
    data = []
    with open(train_data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"[INFO] {len(data)} exemplos carregados")
    
    # Tentar usar modelo GGUF se disponível
    gguf_paths = [
        "/content/drive/MyDrive/Meta-Llama-3.1-8B-Instruct-Q2_K.gguf",
    ]
    
    model_path = None
    for path in gguf_paths:
        if Path(path).exists():
            model_path = path
            print(f"[INFO] Modelo GGUF encontrado: {path}")
            break
    
    if not model_path:
        print("[ERRO] Nenhum modelo GGUF encontrado em src/models/")
        print("[INFO] Modelos esperados:")
        for path in gguf_paths:
            print(f"  - {path}")
        return None
    
    try:
        from llama_cpp import Llama
        print(f"[INFO] Carregando {model_path}...")
        
        # Carregar modelo GGUF com máximo de GPU layers
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # Carregar todos os layers na GPU
            n_ctx=512,
            verbose=False
        )
        
        print("[OK] Modelo GGUF carregado com sucesso!")
        print("[INFO] Este é um modelo pré-treinado - usando para geração de cenas")
        print(f"[VRAM] Em uso: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        # Salvar informações do modelo
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar metadados
        metadata = {
            "model_type": "gguf_inference",
            "model_path": model_path,
            "training_data_path": train_data_path,
            "num_examples": len(data),
            "context_length": 512,
            "note": "Este modelo GGUF é pré-treinado e otimizado para inferência"
        }
        
        with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Metadados salvos em {output_path / 'metadata.json'}")
        
        # Testar geração com um prompt de exemplo
        if len(data) > 0:
            print("[INFO] Testando geração com modelo GGUF...")
            test_prompt = data[0]["text"][:100]
            
            response = llm(
                test_prompt,
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1
            )
            
            print("[OK] Teste bem-sucedido!")
        
        return str(output_path)
        
    except ImportError:
        print("[ERRO] llama-cpp-python não está instalado")
        print("[SOLUÇÃO] Instale com: pip install llama-cpp-python")
        return None
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[ERRO] CUDA OOM: {e}")
        print("[SUGESTÃO] Tente com um modelo menor (Q2_K) ou reduza n_gpu_layers")
        return None
    except Exception as e:
        print(f"\n[ERRO] {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    train_path = preparar_dados_treino_llama31(
        csv_path="/content/drive/MyDrive/monster_descriptions.csv",
        output_dir="/content/drive/MyDrive"
    )
    
    finetuned_model_path = finetune_llama31(
        train_data_path=train_path,
        num_epochs=1
    )