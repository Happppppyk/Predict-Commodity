from __future__ import annotations


def chunk_documents(texts: list[str], chunk_size: int, overlap: int) -> list[str]:
    pass


def embed_chunks(chunks: list[str], model_id: str):
    pass


def upsert_to_vectordb(store, ids: list[str], vectors, metadata: list[dict]) -> None:
    pass


def similarity_search(store, query_vector, top_k: int) -> list[dict]:
    pass
