"""
Sample Agents for AI Agent Framework

This package contains example agents demonstrating framework capabilities.
"""

from .pdf_extractor_agent import (
    create_pdf_extractor_agent,
    create_simple_extractor_agent,
    extract_pdf_text,
    chunk_text,
    text_statistics,
    run_demo
)

__all__ = [
    "create_pdf_extractor_agent",
    "create_simple_extractor_agent",
    "extract_pdf_text",
    "chunk_text",
    "text_statistics",
    "run_demo"
]
