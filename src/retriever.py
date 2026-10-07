import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

class HybridFinancialRetriever:
    """
    Production Hybrid Retriever:
    1. Dense Vector Retrieval (Cosine Similarity via ChromaDB)
    2. Sparse Lexical Retrieval (BM25 for exact financial metrics/tickers)
    3. Reciprocal Rank Fusion (RRF) / Linear Score Combination
    """
    def __init__(self, persist_dir: Path = CHROMA_DIR, top_k: int = 4):
        self.top_k = top_k
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=self.embeddings,
            collection_name="finsecure_corpus"
        )
        self._init_bm25()

    def _init_bm25(self):
        # Fetch all documents to initialize BM25 corpus
        results = self.vector_store.get()
        self.corpus_docs = results["documents"]
        self.corpus_metas = results["metadatas"]
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus_docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k

        # 1. Dense Search
        dense_results = self.vector_store.similarity_search_with_relevance_scores(query, k=k*2)
        dense_scores = {doc.page_content: score for doc, score in dense_results}

        # 2. Sparse BM25 Search
        query_tokens = query.lower().split()
        bm25_scores_list = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores_list) if len(bm25_scores_list) > 0 and max(bm25_scores_list) > 0 else 1.0
        sparse_scores = {
            self.corpus_docs[i]: bm25_scores_list[i] / max_bm25
            for i in range(len(self.corpus_docs))
        }

        # 3. Hybrid Fusion (Dense 0.6 + Sparse 0.4)
        combined_scores = []
        all_doc_texts = set(list(dense_scores.keys()) + list(sparse_scores.keys()))
        for doc_text in all_doc_texts:
            d_score = dense_scores.get(doc_text, 0.0)
            s_score = sparse_scores.get(doc_text, 0.0)
            fusion_score = 0.6 * d_score + 0.4 * s_score
            combined_scores.append((doc_text, fusion_score))

        combined_scores.sort(key=lambda x: x[1], reverse=True)
        top_selected = combined_scores[:k]

        final_docs = []
        for text, score in top_selected:
            # find metadata
            meta = {}
            if text in self.corpus_docs:
                idx = self.corpus_docs.index(text)
                meta = self.corpus_metas[idx]
            final_docs.append({
                "page_content": text,
                "metadata": meta,
                "score": score
            })

        return final_docs

if __name__ == "__main__":
    retriever = HybridFinancialRetriever()
    query = "What was NVIDIA Q3 revenue and Blackwell guidance?"
    results = retriever.retrieve(query, top_k=2)
    print(f"Retrieved {len(results)} docs for query: '{query}'")
    for r in results:
        print(f"Company: {r['metadata'].get('company')} | Score: {r['score']:.4f}")
        print(f"Snippet: {r['page_content'][:150]}...\n")
