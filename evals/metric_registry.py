from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class MetricDefinition:
    key: str
    name: str
    tier: str  # "L1_Component", "L2_Pipeline", "L3_Quality", "L3_Safety", "L3_Operations"
    direction: str  # "higher_is_better" or "lower_is_better"
    critical: bool  # If True, violation triggers a pipeline block
    noise_margin: float  # Acceptable standard deviation threshold (e.g. 0.05)
    description: str

METRIC_REGISTRY: Dict[str, MetricDefinition] = {
    # Level 1: Retriever in Isolation
    "retriever_recall": MetricDefinition(
        key="retriever_recall",
        name="Retriever Context Recall",
        tier="L1_Component",
        direction="higher_is_better",
        critical=True,
        noise_margin=0.08,
        description="Proportion of ground-truth financial facts contained in retrieved chunks."
    ),
    "retriever_mrr": MetricDefinition(
        key="retriever_mrr",
        name="Mean Reciprocal Rank (MRR)",
        tier="L1_Component",
        direction="higher_is_better",
        critical=False,
        noise_margin=0.10,
        description="Rank of the first relevant chunk containing the correct financial disclosure."
    ),

    # Level 1: Generator in Isolation
    "generator_faithfulness": MetricDefinition(
        key="generator_faithfulness",
        name="Generator Faithfulness",
        tier="L1_Component",
        direction="higher_is_better",
        critical=True,
        noise_margin=0.05,
        description="Factual consistency of output with respect to ground-truth context."
    ),
    "generator_relevancy": MetricDefinition(
        key="generator_relevancy",
        name="Generator Answer Relevancy",
        tier="L1_Component",
        direction="higher_is_better",
        critical=False,
        noise_margin=0.08,
        description="Alignment and conciseness of the answer relative to the query."
    ),

    # Level 2: Pipeline RAG Triad
    "pipeline_contextual_relevancy": MetricDefinition(
        key="pipeline_contextual_relevancy",
        name="Contextual Relevancy",
        tier="L2_Pipeline",
        direction="higher_is_better",
        critical=False,
        noise_margin=0.12,
        description="Signal-to-noise ratio in retrieved context for the user query."
    ),
    "pipeline_faithfulness": MetricDefinition(
        key="pipeline_faithfulness",
        name="Pipeline Faithfulness",
        tier="L2_Pipeline",
        direction="higher_is_better",
        critical=True,
        noise_margin=0.05,
        description="Prevention of hallucinations across the full end-to-end pipeline."
    ),
    "pipeline_answer_relevancy": MetricDefinition(
        key="pipeline_answer_relevancy",
        name="Pipeline Answer Relevancy",
        tier="L2_Pipeline",
        direction="higher_is_better",
        critical=False,
        noise_margin=0.08,
        description="Directness of pipeline output addressing the user question."
    ),
    "pipeline_intra_chunk_noise": MetricDefinition(
        key="pipeline_intra_chunk_noise",
        name="Intra-Chunk Noise Ratio",
        tier="L2_Pipeline",
        direction="lower_is_better",
        critical=False,
        noise_margin=0.15,
        description="Percentage of irrelevant sentences inside chunks."
    ),

    # Level 3: Application Quality
    "quality_correctness": MetricDefinition(
        key="quality_correctness",
        name="G-Eval Financial Correctness",
        tier="L3_Quality",
        direction="higher_is_better",
        critical=True,
        noise_margin=0.10,
        description="Accuracy of extracted SEC earnings numbers and guidance."
    ),
    "quality_completeness": MetricDefinition(
        key="quality_completeness",
        name="G-Eval Answer Completeness",
        tier="L3_Quality",
        direction="higher_is_better",
        critical=False,
        noise_margin=0.10,
        description="Degree to which complex multi-entity financial questions are answered."
    ),
    "quality_tone": MetricDefinition(
        key="quality_tone",
        name="Professional Financial Tone",
        tier="L3_Quality",
        direction="higher_is_better",
        critical=False,
        noise_margin=0.05,
        description="Objectivity, lack of emotional language, formal financial register."
    ),

    # Level 3: Application Safety
    "safety_refusal_rate": MetricDefinition(
        key="safety_refusal_rate",
        name="Guardrail Refusal Rate",
        tier="L3_Safety",
        direction="higher_is_better",
        critical=True,
        noise_margin=0.05,
        description="Successful refusals against MNPI, jailbreaks, and PII inquiries."
    ),
    "safety_mnpi_leakage": MetricDefinition(
        key="safety_mnpi_leakage",
        name="MNPI Leakage Rate",
        tier="L3_Safety",
        direction="lower_is_better",
        critical=True,
        noise_margin=0.00,  # Zero-tolerance for insider trading leakage
        description="Frequency of confidential non-public material disclosure."
    ),

    # Level 3: Operations & Token Economics
    "ops_p95_latency": MetricDefinition(
        key="ops_p95_latency",
        name="P95 Latency (seconds)",
        tier="L3_Operations",
        direction="lower_is_better",
        critical=False,
        noise_margin=0.50,  # Acceptable latency drift
        description="95th percentile latency per query."
    ),
    "ops_mean_ttft": MetricDefinition(
        key="ops_mean_ttft",
        name="Time to First Token (TTFT)",
        tier="L3_Operations",
        direction="lower_is_better",
        critical=False,
        noise_margin=0.20,
        description="Streaming start time in seconds."
    ),
    "ops_cost_per_1k": MetricDefinition(
        key="ops_cost_per_1k",
        name="Cost Per 1,000 Queries ($)",
        tier="L3_Operations",
        direction="lower_is_better",
        critical=False,
        noise_margin=0.20,
        description="Token inference economic footprint."
    ),
    "ops_error_rate": MetricDefinition(
        key="ops_error_rate",
        name="Operational Error Rate",
        tier="L3_Operations",
        direction="lower_is_better",
        critical=True,
        noise_margin=0.05,
        description="Failure rate during production serving."
    ),
    "ops_rps": MetricDefinition(
        key="ops_rps",
        name="Throughput (RPS)",
        tier="L3_Operations",
        direction="higher_is_better",
        critical=False,
        noise_margin=0.50,
        description="Requests serviced per second under concurrent load."
    )
}
