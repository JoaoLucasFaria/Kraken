from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from pathlib import Path


class FinetunedModel:
    """Carrega modelo fine-tuned (base HF + adapter LoRA local)"""
    
    def __init__(self, base_model: str = "meta-llama/Llama-3.1-8B-Instruct", 
                 adapter_path: str = "src/models/llama_finetuned",
                 device_map: str = "auto"):
        
        self.adapter_path = Path(adapter_path)
        
        if not self.adapter_path.exists():
            raise FileNotFoundError(f"Adapter não encontrado em {adapter_path}")
        
        print(f"[INFO] Carregando tokenizer do adapter...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path), use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"[INFO] Carregando modelo base: {base_model}...")
        # Carregar modelo base
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            offload_folder="src/models/offload",
            max_memory = {0:"6GiB"},
            load_in_4bit=True,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        print(f"[INFO] Carregando adapter LoRA de {adapter_path}...")
        # Carregar adapter LoRA fine-tunado
        self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path))
        self.model.eval()
        print("[OK] Modelo fine-tuned carregado com sucesso!")
    
    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        """Gera texto usando o modelo fine-tuned"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)