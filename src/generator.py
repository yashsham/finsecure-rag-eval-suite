import os
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

SYSTEM_PROMPT = """You are FinSecure AI, a specialized SEC financial analyst assistant.
Your duty is to answer financial queries strictly and faithfully based on the official transcripts and disclosures provided in the context below.

STRICT OPERATIONAL GUIDELINES:
1. ONLY utilize verifiable facts, figures, and forward guidance directly quoted or stated within the <sec_filing_context>.
2. If the context does not contain sufficient information to answer the question with certainty, state clearly: "I cannot provide an answer because this information is not disclosed in the provided official filing documents."
3. Under NO circumstances disclose Material Non-Public Information (MNPI), personal insider information, rumors, or speculate beyond disclosed numbers.
4. Maintain a strictly professional, neutral, objective financial analyst tone.
5. Cite the company name and any specific revenue or margin percentages where applicable.
"""

USER_TEMPLATE = """<sec_filing_context>
{context}
</sec_filing_context>

<analyst_question>
{question}
</analyst_question>

Provide a rigorous, concise, and faithful financial assessment:"""

class MultiProviderLLM:
    """
    Unified Multi-Provider LLM Orchestrator:
    1. Primary: Groq LPU (qwen/qwen3.8-27b) - Ultra-fast inference
    2. Secondary: NVIDIA NIM (meta/llama-3.2-11b-vision-instruct) - High-throughput GPU fallback
    3. Tertiary: OmniRoute Gateway (openai/gpt-oss-120b or configurable) - Multi-tenant router fallback
    """
    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.nvidia_key = os.getenv("NVIDIA_API_KEY")
        self.omniroute_key = os.getenv("OMNIROUTE_API_KEY")
        self.omniroute_base_url = os.getenv("OMNIROUTE_BASE_URL", "https://api.omniroute.online/v1")

        self.groq_llm = None
        if self.groq_key:
            try:
                self.groq_llm = ChatGroq(
                    model_name="qwen/qwen3.8-27b",
                    groq_api_key=self.groq_key,
                    temperature=temperature
                )
            except Exception as e:
                print(f"[!] Warning: Could not initialize Groq LLM: {e}")

        self.nvidia_llm = None
        if self.nvidia_key:
            try:
                self.nvidia_llm = ChatOpenAI(
                    model="meta/llama-3.2-11b-vision-instruct",
                    openai_api_key=self.nvidia_key,
                    openai_api_base="https://integrate.api.nvidia.com/v1",
                    temperature=temperature
                )
            except Exception as e:
                print(f"[!] Warning: Could not initialize NVIDIA LLM: {e}")

        self.omniroute_llm = None
        if self.omniroute_key:
            try:
                self.omniroute_llm = ChatOpenAI(
                    model="openai/gpt-oss-120b",
                    openai_api_key=self.omniroute_key,
                    openai_api_base=self.omniroute_base_url,
                    temperature=temperature
                )
            except Exception as e:
                print(f"[!] Warning: Could not initialize OmniRoute LLM: {e}")

    def invoke(self, messages, **kwargs):
        errors = []

        # 1. Try Primary: Groq
        if self.groq_llm:
            try:
                return self.groq_llm.invoke(messages, **kwargs)
            except Exception as e:
                print(f"[MultiProvider] Groq invocation failed ({e}). Attempting fallback to NVIDIA NIM...")
                errors.append(f"Groq: {e}")

        # 2. Try Secondary: NVIDIA NIM
        if self.nvidia_llm:
            try:
                res = self.nvidia_llm.invoke(messages, **kwargs)
                print("[MultiProvider] Successfully served via NVIDIA NIM fallback.")
                return res
            except Exception as e:
                print(f"[MultiProvider] NVIDIA NIM invocation failed ({e}). Attempting fallback to OmniRoute...")
                errors.append(f"NVIDIA: {e}")

        # 3. Try Tertiary: OmniRoute
        if self.omniroute_llm:
            try:
                res = self.omniroute_llm.invoke(messages, **kwargs)
                print("[MultiProvider] Successfully served via OmniRoute fallback.")
                return res
            except Exception as e:
                print(f"[MultiProvider] OmniRoute invocation failed ({e}).")
                errors.append(f"OmniRoute: {e}")

        raise RuntimeError(f"All LLM providers failed. Errors: {'; '.join(errors)}")

    def stream(self, messages, **kwargs):
        if self.groq_llm:
            try:
                yield from self.groq_llm.stream(messages, **kwargs)
                return
            except Exception as e:
                print(f"[MultiProvider] Groq stream failed ({e}). Falling back to NVIDIA NIM...")

        if self.nvidia_llm:
            try:
                yield from self.nvidia_llm.stream(messages, **kwargs)
                return
            except Exception as e:
                print(f"[MultiProvider] NVIDIA stream failed ({e}). Falling back to OmniRoute...")

        if self.omniroute_llm:
            try:
                yield from self.omniroute_llm.stream(messages, **kwargs)
                return
            except Exception as e:
                print(f"[MultiProvider] OmniRoute stream failed ({e}).")

        raise RuntimeError("All LLM streaming providers failed.")

class FinancialGenerator:
    """
    Financial Generator with Multi-Provider Orchestration (Groq + NVIDIA NIM + OmniRoute).
    """
    def __init__(self, model_name: str = "hybrid-multi-provider", temperature: float = 0.0):
        self.model_name = model_name
        self.llm = MultiProviderLLM(temperature=temperature)

    def generate(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        context_str = "\n\n---\n\n".join([
            f"[Source: {d.get('metadata', {}).get('company', 'Unknown')} {d.get('metadata', {}).get('quarter', '')} {d.get('metadata', {}).get('fiscal_year', '')}]\n{d.get('page_content', '')}"
            for d in retrieved_docs
        ])

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_TEMPLATE.format(context=context_str, question=question))
        ]

        response = self.llm.invoke(messages)
        return response.content

    def stream_generate(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> Generator[str, None, None]:
        context_str = "\n\n---\n\n".join([
            f"[Source: {d.get('metadata', {}).get('company', 'Unknown')} {d.get('metadata', {}).get('quarter', '')} {d.get('metadata', {}).get('fiscal_year', '')}]\n{d.get('page_content', '')}"
            for d in retrieved_docs
        ])

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_TEMPLATE.format(context=context_str, question=question))
        ]

        for chunk in self.llm.stream(messages):
            yield chunk.content

if __name__ == "__main__":
    generator = FinancialGenerator()
    dummy_docs = [{
        "page_content": "NVIDIA reported record revenue for the third quarter of fiscal 2025 of $35.1 billion, up 94% year-over-year. Data Center revenue was a record $30.8 billion.",
        "metadata": {"company": "NVIDIA", "quarter": "Q3", "fiscal_year": "2025"}
    }]
    ans = generator.generate("What was NVIDIA Q3 revenue and growth rate?", dummy_docs)
    print("Answer:\n", ans)
