import os
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv

from langchain_groq import ChatGroq
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

class FinancialGenerator:
    """
    Financial Generator powered by Groq (qwen/qwen3.8-27b) with negative financial constraints.
    """
    def __init__(self, model_name: str = "qwen/qwen3.8-27b", temperature: float = 0.0):
        self.model_name = model_name
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY must be set.")
        self.llm = ChatGroq(
            model_name=model_name,
            groq_api_key=api_key,
            temperature=temperature
        )

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
