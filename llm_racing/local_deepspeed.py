import os

from llm_racing.racer import LLMRacer
import deepspeed
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import torch

class DeepSpeedRacer(LLMRacer):
    def __init__(self, model: str = "EleutherAI/pythia-70m-deduped"):
        super().__init__(model)
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        world_size = int(os.getenv('WORLD_SIZE', '1'))
        self.device = torch.device(f'cuda:{local_rank}')
        self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(model)
        self.model = GPTNeoXForCausalLM.from_pretrained(model)

        self.model = deepspeed.init_inference(
            self.model, 
            mp_size=world_size, 
            dtype=torch.float16, 
            checkpoint=None,
            replace_method='auto',
            replace_with_kernel_inject=True,
        )


    def generate_tokens(self, prompt: str, max_tokens: int, min_tokens: int = 0) -> int:
        tokenized_inputs = self.tokenizer(
            list(prompt),
            return_tensors='pt', 
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=tokenized_inputs['input_ids'],
                attention_mask=tokenized_inputs['attention_mask'],
                use_cache=True,
                do_sample=False, 
                min_length=min_tokens, 
                max_length=max_tokens
            )
        return outputs[0].nelement()