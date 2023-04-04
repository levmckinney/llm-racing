import abc
from typing import List, Optional
import pandas as pd
import time
from tqdm import tqdm

class LLMRacer(abc.ABC):

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abc.abstractmethod
    def generate_tokens(self, prompt: str, max_tokens: int, min_tokens: int = 0) -> int:
        """Generate tokens from prompt.

        Args:
            prompt: Prompt to generate tokens from.
            max_tokens: Maximum number of tokens to generate.
        
        Returns:
            Number of tokens generated.
        """
        ...

    def time_trial(self, prompts: List[str], max_tokens: int, target_tokens: Optional[int] = None) -> pd.DataFrame:
        """Generate tokens and time it.

        Args:
            prompts: the set of prompts to time.
            max_tokens: Maximum number of tokens to generate.
            target_tokens: Target number of tokens to generate.

        Returns:
            DataFrame where each row is a sample and the columns are:
                - `tokens`: Number of tokens generated.
                - `time`: Time taken to generate tokens.
        """
        results = []
        target_tokens = target_tokens if target_tokens else [None] * len(prompts)
        print("running once to warm up")
        self.generate_tokens(prompts[0], max_tokens=max_tokens, min_tokens=0)

        print("Starting time trial for", self.model_name)
        for prompt, target in tqdm(list(zip(prompts, target_tokens))):

            start_time = time.time()
            generated_tokens = self.generate_tokens(
                prompt, 
                max_tokens=min(target, max_tokens), 
                min_tokens=target - 1 if target is not None else 0
            )
            end_time = time.time()


            elapsed_time = end_time - start_time
            results.append({"tokens": generated_tokens, "time": elapsed_time, "model_name": self.model_name})

        return pd.DataFrame(results)