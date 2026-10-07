import os
import time
import asyncio
from typing import Optional, Tuple
from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq

load_dotenv()

class GroqJudgeModel(DeepEvalBaseLLM):
    """
    Production DeepEval Judge wrapper powered by Groq with rate-limit resilient retry & backoff.
    Default model: qwen/qwen3.8-27b
    """
    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        self.model_name = model_name
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
        self.llm = ChatGroq(model_name=self.model_name, groq_api_key=api_key, temperature=0.0)
        super().__init__(model_name=self.model_name)

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        max_retries = 5
        base_delay = 5.0
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                return response.content
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait_time = base_delay * (attempt + 1)
                    print(f"[!] Groq Judge 429 rate limit reached. Pausing {wait_time}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise e
        # Final attempt
        return self.llm.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        max_retries = 5
        base_delay = 5.0
        for attempt in range(max_retries):
            try:
                response = await self.llm.ainvoke(prompt)
                return response.content
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait_time = base_delay * (attempt + 1)
                    print(f"[!] Groq Judge async 429 rate limit reached. Pausing {wait_time}s before retry {attempt+1}/{max_retries}...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e
        # Final attempt
        res = await self.llm.ainvoke(prompt)
        return res.content

    def get_model_name(self) -> str:
        return self.model_name
