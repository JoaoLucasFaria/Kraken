from llama_cpp import Llama, CreateChatCompletionResponse
from typing import cast

class LocalModel:
    """ Depois transformar a entrada do construtor em um objeto """
    def __init__(self, model_path: str, n_ctx: int = 1024, temperature: float = 0.7, top_p: float = 0.9, repeat_penalty: float = 1.1):
        self.temperature = temperature
        self.top_p = top_p 
        self.repeat_penalty = repeat_penalty

        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=8, verbose=False)

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        result: CreateChatCompletionResponse = cast(CreateChatCompletionResponse, self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
            stop=["###", "<|end|>", "</s>"],
            stream=False
        ))
        res: str | None = result["choices"][0]["message"]["content"] 
        return res if res is not None else ""
