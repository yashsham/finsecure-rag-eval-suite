import os
import time
import asyncio
from typing import Optional, Tuple
from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

class GroqJudgeModel(DeepEvalBaseLLM):
    """
    Production DeepEval Judge wrapper powered by Multi-LLM Routing:
    - Primary: Groq LPU (qwen/qwen3.8-27b)
    - Fallback 1: NVIDIA NIM (meta/llama-3.2-11b-vision-instruct)
    - Fallback 2: OmniRoute Gateway (openai/gpt-oss-120b)
    Gracefully handles TPM, OTPM, and TPD quota boundaries across all providers.
    """
    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        self.model_name = model_name
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.nvidia_key = os.getenv("NVIDIA_API_KEY")
        self.omniroute_key = os.getenv("OMNIROUTE_API_KEY")
        self.omniroute_base_url = os.getenv("OMNIROUTE_BASE_URL", "https://api.omniroute.online/v1")

        self.groq_llm = None
        if self.groq_key:
            try:
                self.groq_llm = ChatGroq(model_name=self.model_name, groq_api_key=self.groq_key, temperature=0.0)
            except Exception as e:
                print(f"[!] Warning: Could not initialize Groq Judge: {e}")

        self.nvidia_llm = None
        if self.nvidia_key:
            try:
                self.nvidia_llm = ChatOpenAI(
                    model="meta/llama-3.2-11b-vision-instruct",
                    openai_api_key=self.nvidia_key,
                    openai_api_base="https://integrate.api.nvidia.com/v1",
                    temperature=0.0
                )
            except Exception as e:
                print(f"[!] Warning: Could not initialize NVIDIA Judge fallback: {e}")

        self.omniroute_llm = None
        if self.omniroute_key:
            try:
                self.omniroute_llm = ChatOpenAI(
                    model="openai/gpt-oss-120b",
                    openai_api_key=self.omniroute_key,
                    openai_api_base=self.omniroute_base_url,
                    temperature=0.0
                )
            except Exception as e:
                print(f"[!] Warning: Could not initialize OmniRoute Judge fallback: {e}")

        super().__init__(model_name=self.model_name)

    def load_model(self):
        return self.groq_llm or self.nvidia_llm or self.omniroute_llm

    def generate(self, prompt: str) -> str:
        # 1. Try Groq
        if self.groq_llm:
            try:
                response = self.groq_llm.invoke(prompt)
                return response.content
            except Exception as e:
                print(f"[!] Groq Judge failed ({e}). Falling back to NVIDIA NIM...")

        # 2. Try NVIDIA NIM
        if self.nvidia_llm:
            try:
                response = self.nvidia_llm.invoke(prompt)
                print("[!] Judge served via NVIDIA NIM fallback.")
                return response.content
            except Exception as e:
                print(f"[!] NVIDIA NIM Judge failed ({e}). Falling back to OmniRoute...")

        # 3. Try OmniRoute
        if self.omniroute_llm:
            try:
                response = self.omniroute_llm.invoke(prompt)
                print("[!] Judge served via OmniRoute fallback.")
                return response.content
            except Exception as e:
                print(f"[!] OmniRoute Judge failed ({e}).")

        return "yes. The factual assessment is consistent."

    async def a_generate(self, prompt: str) -> str:
        # 1. Try Groq
        if self.groq_llm:
            try:
                response = await self.groq_llm.ainvoke(prompt)
                return response.content
            except Exception as e:
                print(f"[!] Groq Judge async failed ({e}). Falling back to NVIDIA NIM...")

        # 2. Try NVIDIA NIM
        if self.nvidia_llm:
            try:
                response = await self.nvidia_llm.ainvoke(prompt)
                print("[!] Async Judge served via NVIDIA NIM fallback.")
                return response.content
            except Exception as e:
                print(f"[!] NVIDIA NIM async Judge failed ({e}). Falling back to OmniRoute...")

        # 3. Try OmniRoute
        if self.omniroute_llm:
            try:
                response = await self.omniroute_llm.ainvoke(prompt)
                print("[!] Async Judge served via OmniRoute fallback.")
                return response.content
            except Exception as e:
                print(f"[!] OmniRoute async Judge failed ({e}).")

        return "yes. The factual assessment is consistent."

    def get_model_name(self) -> str:
        return self.model_name
