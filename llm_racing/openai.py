from llm_racing.racer import LLMRacer
import openai


class OpenAICompletionsRacer(LLMRacer):

    def __init__(self, engine: str = "text-davinci-002"):
        super().__init__(engine)
        self.engine = engine

    def generate_tokens(self, prompt: str, max_tokens: int) -> int:
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=1.0,
        )

        generated_tokens = response.usage.completion_tokens

        return generated_tokens


class OpenAIChatRacer(LLMRacer):

    def __init__(self, engine: str = "gpt-3.5-turbo"):
        super().__init__(engine)
        self.engine = engine

    def generate_tokens(self, prompt: str, max_tokens: int, min_tokens: int = 0) -> int:
        del min_tokens
        response = openai.ChatCompletion.create(
            model=self.engine,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=1.0,
        )

        generated_tokens = response.usage.completion_tokens

        return generated_tokens