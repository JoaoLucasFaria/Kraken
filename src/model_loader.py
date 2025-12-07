from llama_cpp import Llama, CreateChatCompletionResponse
from typing import cast
import torch

class LocalModel:
    """ Carrega modelo GGUF localmente """
    def __init__(self, model_path: str, n_ctx: int = 1024, temperature: float = 0.7, top_p: float = 0.9, repeat_penalty: float = 1.1, n_gpu_layers: int = None):
        self.temperature = temperature
        self.top_p = top_p 
        self.repeat_penalty = repeat_penalty

        # Auto-detectar GPU se não especificado
        if n_gpu_layers is None:
            # Q2_K é muito leve, pode usar até 35 camadas na GPU
            n_gpu_layers = 35 if torch.cuda.is_available() else 0

        # Carregar modelo GGUF
        self.llm = Llama(
            model_path=model_path, 
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False
        )

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        result: CreateChatCompletionResponse = cast(CreateChatCompletionResponse, self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
            stop=["###", "<|end|>", "</s>", "<|eot_id|>"],
            stream=False
        ))
        res: str | None = result["choices"][0]["message"]["content"] 
        return res if res is not None else ""
