"""
역할: 정책 문서·리포트·FAQ 등을 임베딩하여 벡터 DB에 적재·검색한다.
레이어: RAG (6) — LLM 컨텍스트 주입용 retrieval.
"""

# 핵심 import (구현 시 사용 예정)
# from pathlib import Path
# import numpy as np


def chunk_documents(texts: list[str], chunk_size: int, overlap: int) -> list[str]:
    """긴 문서를 검색 단위 청크로 분할한다."""
    pass


def embed_chunks(chunks: list[str], model_id: str):
    """청크 임베딩 벡터를 반환한다."""
    pass


def upsert_to_vectordb(store, ids: list[str], vectors, metadata: list[dict]) -> None:
    """벡터 스토어(로컬 SQLite 확장, Chroma, FAISS 등)에 upsert한다."""
    pass


def similarity_search(store, query_vector, top_k: int) -> list[dict]:
    """쿼리 벡터와 유사한 문서 청크를 반환한다."""
    pass
