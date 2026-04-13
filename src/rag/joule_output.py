from __future__ import annotations


def build_prompt(system_instructions: str, retrieved_docs: list[str], structured_facts: dict) -> str:
    pass


def call_llm(prompt: str, model_config: dict) -> str:
    pass


def format_joule_response(natural_language: str, citations: list[dict], metrics: dict) -> dict:
    pass


def validate_joule_schema(payload: dict) -> bool:
    pass
