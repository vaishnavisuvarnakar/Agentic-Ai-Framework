"""
PDF to Text Extractor Agent

A simple agent that extracts text from PDF files and processes it using Google Gemini API.
This agent demonstrates the AI Agent Framework capabilities for document processing workflows.

Usage:
    from sampleagents.pdf_extractor_agent import create_pdf_extractor_agent
    
    agent = create_pdf_extractor_agent(api_key="your-gemini-api-key")
    result = agent.run_flow("pdf_extraction_workflow", context={
        "pdf_path": "path/to/document.pdf",
        "query": "Summarize this document"
    })
"""

import os
import time
from typing import Any, Dict, List, Optional

# Import from our framework
from framework import (
    Agent,
    Flow,
    FunctionTask,
    tool,
    tool_registry,
    LogLevel,
    setup_logging,
    LLMTool
)

# Gemini API support
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


# =============================================================================
# PDF EXTRACTION TOOLS
# =============================================================================

@tool(
    name="extract_pdf_text",
    description="Extract text content from a PDF file",
    parameters={
        "pdf_path": {"type": "string", "description": "Path to the PDF file"},
        "page_range": {"type": "string", "description": "Optional page range (e.g., '1-5' or 'all')", "required": False}
    },
    tags=["pdf", "extraction", "document"]
)
def extract_pdf_text(pdf_path: str, page_range: str = "all") -> Dict[str, Any]:
    """
    Extract text from a PDF file using PyMuPDF (fitz).
    
    Args:
        pdf_path: Path to the PDF file
        page_range: Page range to extract (e.g., '1-5', 'all')
        
    Returns:
        Dictionary containing extracted text and metadata
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "PyMuPDF not installed. Install with: pip install PyMuPDF"
        )
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # Parse page range
    if page_range == "all":
        start_page, end_page = 0, total_pages
    else:
        try:
            parts = page_range.split("-")
            start_page = int(parts[0]) - 1  # Convert to 0-indexed
            end_page = int(parts[1]) if len(parts) > 1 else start_page + 1
        except (ValueError, IndexError):
            start_page, end_page = 0, total_pages
    
    # Extract text from each page
    pages_text = []
    full_text = []
    
    for page_num in range(max(0, start_page), min(end_page, total_pages)):
        page = doc[page_num]
        text = page.get_text()
        pages_text.append({
            "page_number": page_num + 1,
            "text": text,
            "char_count": len(text)
        })
        full_text.append(text)
    
    doc.close()
    
    combined_text = "\n\n".join(full_text)
    
    return {
        "pdf_path": pdf_path,
        "total_pages": total_pages,
        "extracted_pages": len(pages_text),
        "page_range": f"{start_page + 1}-{end_page}",
        "full_text": combined_text,
        "pages": pages_text,
        "total_characters": len(combined_text),
        "total_words": len(combined_text.split())
    }


@tool(
    name="chunk_text",
    description="Split text into smaller chunks for processing",
    parameters={
        "text": {"type": "string", "description": "Text to chunk"},
        "chunk_size": {"type": "integer", "description": "Maximum characters per chunk", "required": False},
        "overlap": {"type": "integer", "description": "Overlap between chunks", "required": False}
    },
    tags=["text", "processing", "chunking"]
)
def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> Dict[str, Any]:
    """
    Split text into smaller chunks suitable for LLM processing.
    
    Args:
        text: Text to split
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks
        
    Returns:
        Dictionary containing chunks and metadata
    """
    if not text:
        return {"chunks": [], "total_chunks": 0}
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence or paragraph boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break
            else:
                # Look for sentence break
                for punct in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                    sent_break = text.rfind(punct, start, end)
                    if sent_break > start + chunk_size // 2:
                        end = sent_break + len(punct)
                        break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "index": len(chunks),
                "text": chunk,
                "start_pos": start,
                "end_pos": end,
                "char_count": len(chunk)
            })
        
        start = end - overlap if end < len(text) else len(text)
    
    return {
        "chunks": chunks,
        "total_chunks": len(chunks),
        "total_characters": len(text),
        "chunk_size": chunk_size,
        "overlap": overlap
    }


@tool(
    name="text_statistics",
    description="Get statistics about the extracted text",
    parameters={
        "text": {"type": "string", "description": "Text to analyze"}
    },
    tags=["text", "analysis", "statistics"]
)
def text_statistics(text: str) -> Dict[str, Any]:
    """
    Calculate basic statistics about the text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary containing text statistics
    """
    if not text:
        return {
            "characters": 0,
            "words": 0,
            "sentences": 0,
            "paragraphs": 0,
            "lines": 0
        }
    
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    paragraphs = text.count('\n\n') + 1
    lines = text.count('\n') + 1
    
    # Estimate reading time (average 200 words per minute)
    reading_time_minutes = len(words) / 200
    
    return {
        "characters": len(text),
        "characters_no_spaces": len(text.replace(" ", "").replace("\n", "")),
        "words": len(words),
        "sentences": sentences,
        "paragraphs": paragraphs,
        "lines": lines,
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        "avg_sentence_length": len(words) / sentences if sentences else 0,
        "reading_time_minutes": round(reading_time_minutes, 1)
    }


# =============================================================================
# GEMINI LLM WRAPPER
# =============================================================================

class GeminiLLM:
    """
    Wrapper for Google Gemini API from AI Studio.
    
    Uses the google-generativeai library to communicate with Gemini models.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        max_tokens: int = 2048,
        temperature: float = 0.3
    ):
        """
        Initialize Gemini LLM.
        
        Args:
            api_key: Google AI Studio API key
            model: Gemini model name (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro')
            max_tokens: Maximum output tokens
            temperature: Sampling temperature (0-1)
        """
        if not GEMINI_AVAILABLE:
            raise RuntimeError(
                "google-generativeai not installed. Install with: pip install google-generativeai"
            )
        
        self.api_key = api_key
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
        )
    
    def generate(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        Generate a response from Gemini.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Build the full prompt with system instruction
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            return {
                "response": response.text,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0
                }
            }
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "model": self.model_name,
                "error": str(e)
            }
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute method for compatibility with the framework.
        
        Args:
            inputs: Dictionary with 'prompt' and optional 'system_prompt'
            
        Returns:
            Generation result
        """
        prompt = inputs.get("prompt", "")
        system_prompt = inputs.get("system_prompt", None)
        return self.generate(prompt, system_prompt)


# =============================================================================
# OPENROUTER LLM WRAPPER (for DeepSeek and other models)
# =============================================================================

class OpenRouterLLM:
    """
    Wrapper for OpenRouter API - supports DeepSeek and many other models.
    
    OpenRouter provides access to various LLM models including free DeepSeek models.
    Uses OpenAI-compatible API format.
    """
    
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek/deepseek-chat-free",
        max_tokens: int = 2048,
        temperature: float = 0.3
    ):
        """
        Initialize OpenRouter LLM.
        
        Args:
            api_key: OpenRouter API key
            model: Model name (e.g., 'deepseek/deepseek-chat-free', 'deepseek/deepseek-r1-0528')
            max_tokens: Maximum output tokens
            temperature: Sampling temperature (0-1)
        """
        self.api_key = api_key
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Try to import openai library
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.OPENROUTER_BASE_URL
            )
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Install with: pip install openai"
            )
    
    def generate(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        Generate a response from OpenRouter.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return {
                "response": response.choices[0].message.content,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0
                }
            }
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "model": self.model_name,
                "error": str(e)
            }
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute method for compatibility with the framework.
        
        Args:
            inputs: Dictionary with 'prompt' and optional 'system_prompt'
            
        Returns:
            Generation result
        """
        prompt = inputs.get("prompt", "")
        system_prompt = inputs.get("system_prompt", None)
        return self.generate(prompt, system_prompt)


# =============================================================================
# PDF EXTRACTOR AGENT WITH OPENROUTER
# =============================================================================

def create_pdf_extractor_agent_openrouter(
    api_key: Optional[str] = None,
    model: str = "deepseek/deepseek-chat-free",
    max_tokens: int = 2048,
    temperature: float = 0.3
) -> Agent:
    """
    Create a PDF to Text Extractor Agent using OpenRouter API (DeepSeek).
    
    This agent provides a workflow to:
    1. Extract text from PDF files
    2. Analyze the text statistics
    3. Process/summarize using OpenRouter LLM (DeepSeek, etc.)
    
    Args:
        api_key: OpenRouter API key (or set via OPEN_ROUTER_API env var)
        model: Model name (default: 'deepseek/deepseek-chat-free')
               Other options: 'deepseek/deepseek-r1-0528', 'meta-llama/llama-3-8b-instruct:free'
        max_tokens: Maximum output tokens for LLM response
        temperature: Sampling temperature (0-1)
        
    Returns:
        Configured Agent instance
    """
    # Get API key from parameter or environment
    api_key = api_key or os.environ.get("OPEN_ROUTER_API")
    
    if not api_key:
        raise ValueError(
            "API key required. Pass api_key parameter or set OPEN_ROUTER_API environment variable. "
            "Get your key from: https://openrouter.ai/keys"
        )
    
    agent = Agent(
        name="PDFExtractorAgent",
        description="An agent that extracts and processes text from PDF documents using OpenRouter (DeepSeek)"
    )
    
    # Create the OpenRouter LLM instance
    llm = OpenRouterLLM(
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # Create the main extraction workflow
    flow = agent.create_flow(
        name="pdf_extraction_workflow",
        description="Extract text from PDF and process with LLM",
        max_workers=2
    )
    
    # Task 1: Extract text from PDF
    extract_task = FunctionTask(
        name="extract_text",
        func=lambda ctx: tool_registry.execute(
            "extract_pdf_text",
            {
                "pdf_path": ctx.get("pdf_path"),
                "page_range": ctx.get("page_range", "all")
            }
        ),
        description="Extract text content from the PDF file",
        max_retries=2
    )
    
    # Task 2: Get text statistics
    stats_task = FunctionTask(
        name="analyze_stats",
        func=lambda ctx: tool_registry.execute(
            "text_statistics",
            {"text": ctx.get("extract_text_result", {}).get("full_text", "")}
        ),
        description="Calculate text statistics from extracted content"
    )
    
    # Task 3: Chunk text if needed (for large documents)
    chunk_task = FunctionTask(
        name="chunk_text",
        func=lambda ctx: tool_registry.execute(
            "chunk_text",
            {
                "text": ctx.get("extract_text_result", {}).get("full_text", ""),
                "chunk_size": ctx.get("chunk_size", 4000),
                "overlap": ctx.get("overlap", 200)
            }
        ),
        description="Split text into chunks for LLM processing"
    )
    
    # Task 4: Process with OpenRouter LLM
    llm_task = FunctionTask(
        name="llm_process",
        func=lambda ctx: _process_with_openrouter(ctx, llm),
        description="Process the extracted text using OpenRouter LLM"
    )
    
    # Task 5: Generate final report
    report_task = FunctionTask(
        name="generate_report",
        func=lambda ctx: {
            "pdf_path": ctx.get("pdf_path"),
            "extraction": {
                "total_pages": ctx.get("extract_text_result", {}).get("total_pages"),
                "extracted_pages": ctx.get("extract_text_result", {}).get("extracted_pages"),
                "total_characters": ctx.get("extract_text_result", {}).get("total_characters"),
                "total_words": ctx.get("extract_text_result", {}).get("total_words")
            },
            "statistics": ctx.get("analyze_stats_result", {}),
            "chunks": ctx.get("chunk_text_result", {}).get("total_chunks", 0),
            "llm_response": ctx.get("llm_process_result", {}).get("response", ""),
            "query": ctx.get("query", ""),
            "model_used": ctx.get("llm_process_result", {}).get("model", ""),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        description="Generate final processing report"
    )
    
    # Add tasks to flow
    flow.add_tasks(extract_task, stats_task, chunk_task, llm_task, report_task)
    
    # Define dependencies
    flow.add_dependency("analyze_stats", "extract_text")
    flow.add_dependency("chunk_text", "extract_text")
    flow.add_dependency("llm_process", "chunk_text")
    flow.add_dependency("generate_report", "analyze_stats")
    flow.add_dependency("generate_report", "llm_process")
    
    return agent


def _process_with_openrouter(ctx: Dict[str, Any], llm: OpenRouterLLM) -> Dict[str, Any]:
    """
    Process extracted text with OpenRouter LLM.
    
    Args:
        ctx: Execution context containing extracted text and query
        llm: Configured OpenRouterLLM instance
        
    Returns:
        LLM processing result
    """
    chunks = ctx.get("chunk_text_result", {}).get("chunks", [])
    query = ctx.get("query", "Summarize this document")
    
    if not chunks:
        return {"response": "No text content to process", "model": "none"}
    
    # For simplicity, use first chunk or combine if small enough
    if len(chunks) == 1:
        text_to_process = chunks[0]["text"]
    else:
        # Combine first few chunks up to reasonable size
        combined = []
        total_chars = 0
        max_chars = 8000  # Leave room for prompt and response
        
        for chunk in chunks:
            if total_chars + chunk["char_count"] <= max_chars:
                combined.append(chunk["text"])
                total_chars += chunk["char_count"]
            else:
                break
        
        text_to_process = "\n\n".join(combined)
        if len(chunks) > len(combined):
            text_to_process += f"\n\n[Note: Document truncated. Showing {len(combined)} of {len(chunks)} chunks]"
    
    # Build the prompt
    prompt = f"""Here is the extracted text from a PDF document:

---
{text_to_process}
---

User Query: {query}

Please respond to the user's query based on the document content above."""

    system_prompt = """You are a helpful document analysis assistant. 
Analyze the provided document text and respond to user queries accurately and concisely.
If asked to summarize, provide a clear and structured summary.
If asked questions, answer based on the document content."""

    try:
        result = llm.execute({
            "prompt": prompt,
            "system_prompt": system_prompt
        })
        return result
    except Exception as e:
        return {
            "response": f"Error processing with OpenRouter: {str(e)}",
            "model": llm.model_name,
            "error": str(e)
        }


# =============================================================================
# PDF EXTRACTOR AGENT
# =============================================================================

def create_pdf_extractor_agent(
    api_key: Optional[str] = None,
    model: str = "gemini-1.5-flash",
    max_tokens: int = 2048,
    temperature: float = 0.3
) -> Agent:
    """
    Create a PDF to Text Extractor Agent using Google Gemini API.
    
    This agent provides a workflow to:
    1. Extract text from PDF files
    2. Analyze the text statistics
    3. Process/summarize using Google Gemini LLM
    
    Args:
        api_key: Google AI Studio API key (or set via GOOGLE_API_KEY env var)
        model: Gemini model name ('gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash')
        max_tokens: Maximum output tokens for LLM response
        temperature: Sampling temperature (0-1)
        
    Returns:
        Configured Agent instance
    """
    # Get API key from parameter or environment
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError(
            "API key required. Pass api_key parameter or set GOOGLE_API_KEY environment variable. "
            "Get your key from: https://aistudio.google.com/apikey"
        )
    
    agent = Agent(
        name="PDFExtractorAgent",
        description="An agent that extracts and processes text from PDF documents using Google Gemini"
    )
    
    # Create the Gemini LLM instance
    gemini_llm = GeminiLLM(
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # Create the main extraction workflow
    flow = agent.create_flow(
        name="pdf_extraction_workflow",
        description="Extract text from PDF and process with LLM",
        max_workers=2
    )
    
    # Task 1: Extract text from PDF
    extract_task = FunctionTask(
        name="extract_text",
        func=lambda ctx: tool_registry.execute(
            "extract_pdf_text",
            {
                "pdf_path": ctx.get("pdf_path"),
                "page_range": ctx.get("page_range", "all")
            }
        ),
        description="Extract text content from the PDF file",
        max_retries=2
    )
    
    # Task 2: Get text statistics
    stats_task = FunctionTask(
        name="analyze_stats",
        func=lambda ctx: tool_registry.execute(
            "text_statistics",
            {"text": ctx.get("extract_text_result", {}).get("full_text", "")}
        ),
        description="Calculate text statistics from extracted content"
    )
    
    # Task 3: Chunk text if needed (for large documents)
    chunk_task = FunctionTask(
        name="chunk_text",
        func=lambda ctx: tool_registry.execute(
            "chunk_text",
            {
                "text": ctx.get("extract_text_result", {}).get("full_text", ""),
                "chunk_size": ctx.get("chunk_size", 4000),
                "overlap": ctx.get("overlap", 200)
            }
        ),
        description="Split text into chunks for LLM processing"
    )
    
    # Task 4: Process with Gemini LLM
    llm_task = FunctionTask(
        name="llm_process",
        func=lambda ctx: _process_with_gemini(ctx, gemini_llm),
        description="Process the extracted text using Gemini LLM"
    )
    
    # Task 5: Generate final report
    report_task = FunctionTask(
        name="generate_report",
        func=lambda ctx: {
            "pdf_path": ctx.get("pdf_path"),
            "extraction": {
                "total_pages": ctx.get("extract_text_result", {}).get("total_pages"),
                "extracted_pages": ctx.get("extract_text_result", {}).get("extracted_pages"),
                "total_characters": ctx.get("extract_text_result", {}).get("total_characters"),
                "total_words": ctx.get("extract_text_result", {}).get("total_words")
            },
            "statistics": ctx.get("analyze_stats_result", {}),
            "chunks": ctx.get("chunk_text_result", {}).get("total_chunks", 0),
            "llm_response": ctx.get("llm_process_result", {}).get("response", ""),
            "query": ctx.get("query", ""),
            "model_used": ctx.get("llm_process_result", {}).get("model", ""),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        description="Generate final processing report"
    )
    
    # Add tasks to flow
    flow.add_tasks(extract_task, stats_task, chunk_task, llm_task, report_task)
    
    # Define dependencies (DAG structure):
    #     extract_text
    #     /    |    \
    # stats  chunk  (wait)
    #            \    /
    #          llm_process
    #               |
    #           report
    
    flow.add_dependency("analyze_stats", "extract_text")
    flow.add_dependency("chunk_text", "extract_text")
    flow.add_dependency("llm_process", "chunk_text")
    flow.add_dependency("generate_report", "analyze_stats")
    flow.add_dependency("generate_report", "llm_process")
    
    return agent


def _process_with_gemini(ctx: Dict[str, Any], gemini_llm: GeminiLLM) -> Dict[str, Any]:
    """
    Process extracted text with Google Gemini.
    
    Args:
        ctx: Execution context containing extracted text and query
        gemini_llm: Configured GeminiLLM instance
        
    Returns:
        Gemini processing result
    """
    chunks = ctx.get("chunk_text_result", {}).get("chunks", [])
    query = ctx.get("query", "Summarize this document")
    
    if not chunks:
        return {"response": "No text content to process", "model": "none"}
    
    # For simplicity, use first chunk or combine if small enough
    if len(chunks) == 1:
        text_to_process = chunks[0]["text"]
    else:
        # Combine first few chunks up to reasonable size
        combined = []
        total_chars = 0
        max_chars = 8000  # Leave room for prompt and response
        
        for chunk in chunks:
            if total_chars + chunk["char_count"] <= max_chars:
                combined.append(chunk["text"])
                total_chars += chunk["char_count"]
            else:
                break
        
        text_to_process = "\n\n".join(combined)
        if len(chunks) > len(combined):
            text_to_process += f"\n\n[Note: Document truncated. Showing {len(combined)} of {len(chunks)} chunks]"
    
    # Build the prompt
    prompt = f"""Here is the extracted text from a PDF document:

---
{text_to_process}
---

User Query: {query}

Please respond to the user's query based on the document content above."""

    system_prompt = """You are a helpful document analysis assistant. 
Analyze the provided document text and respond to user queries accurately and concisely.
If asked to summarize, provide a clear and structured summary.
If asked questions, answer based on the document content."""

    try:
        result = gemini_llm.execute({
            "prompt": prompt,
            "system_prompt": system_prompt
        })
        return result
    except Exception as e:
        return {
            "response": f"Error processing with Gemini: {str(e)}",
            "model": gemini_llm.model_name,
            "error": str(e)
        }


# =============================================================================
# SIMPLE EXTRACTION FLOW (without LLM)
# =============================================================================

def create_simple_extractor_agent() -> Agent:
    """
    Create a simple PDF extractor agent that only extracts text (no LLM).
    
    Use this when you just need to extract text from PDFs without LLM processing.
    
    Returns:
        Configured Agent instance
    """
    agent = Agent(
        name="SimplePDFExtractor",
        description="Simple agent that extracts text from PDF documents"
    )
    
    flow = agent.create_flow(
        name="simple_extraction",
        description="Extract text from PDF",
        max_workers=2
    )
    
    # Task 1: Extract text
    extract_task = FunctionTask(
        name="extract",
        func=lambda ctx: tool_registry.execute(
            "extract_pdf_text",
            {
                "pdf_path": ctx.get("pdf_path"),
                "page_range": ctx.get("page_range", "all")
            }
        ),
        description="Extract text from PDF"
    )
    
    # Task 2: Statistics
    stats_task = FunctionTask(
        name="stats",
        func=lambda ctx: tool_registry.execute(
            "text_statistics",
            {"text": ctx.get("extract_result", {}).get("full_text", "")}
        ),
        description="Calculate text statistics"
    )
    
    # Task 3: Output
    output_task = FunctionTask(
        name="output",
        func=lambda ctx: {
            "pdf_path": ctx.get("pdf_path"),
            "text": ctx.get("extract_result", {}).get("full_text", ""),
            "pages": ctx.get("extract_result", {}).get("total_pages", 0),
            "statistics": ctx.get("stats_result", {}),
            "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        description="Output extraction results"
    )
    
    flow.add_tasks(extract_task, stats_task, output_task)
    flow.add_dependency("stats", "extract")
    flow.add_dependency("output", "stats")
    
    return agent


# =============================================================================
# DEMO RUNNER
# =============================================================================

def run_demo(
    pdf_path: str,
    api_key: Optional[str] = None,
    query: str = "Summarize this document in 3-5 bullet points",
    model: str = "gemini-1.5-flash"
):
    """
    Run a demo of the PDF Extractor Agent with Google Gemini.
    
    Args:
        pdf_path: Path to a PDF file
        api_key: Google AI Studio API key (required)
        query: Query to process the document with
        model: Gemini model to use
    """
    print("\n" + "=" * 60)
    print("PDF EXTRACTOR AGENT DEMO (Google Gemini)")
    print("=" * 60)
    
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("\nError: API key required!")
        print("Get your API key from: https://aistudio.google.com/apikey")
        print("Then run: python pdf_extractor_agent.py <pdf_path> <api_key>")
        return None
    
    agent = create_pdf_extractor_agent(
        api_key=api_key,
        model=model
    )
    
    print(f"\nProcessing PDF: {pdf_path}")
    print(f"Query: {query}")
    print(f"Model: {model}")
    
    # Execute the workflow
    result = agent.run_flow(
        "pdf_extraction_workflow",
        context={
            "pdf_path": pdf_path,
            "query": query
        }
    )
    
    print(f"\nFlow Status: {result.status.value}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"Success: {result.success}")
    
    if result.success:
        report = result.task_results.get("generate_report")
        if report and report.output:
            print("\n--- Extraction Report ---")
            output = report.output
            print(f"  PDF: {output.get('pdf_path')}")
            print(f"  Total Pages: {output.get('extraction', {}).get('total_pages')}")
            print(f"  Total Words: {output.get('extraction', {}).get('total_words')}")
            print(f"  Total Chunks: {output.get('chunks')}")
            print(f"  Processed At: {output.get('processed_at')}")
            print(f"\n--- Gemini Response ---")
            print(output.get('llm_response', 'No response'))
    else:
        print(f"\nErrors: {result.errors}")
    
    return result


def main():
    """Main entry point for the demo."""
    import sys
    
    if len(sys.argv) < 2:
        print("PDF to Text Extractor Agent - Using Google Gemini")
        print("=" * 50)
        print("\nUsage: python pdf_extractor_agent.py <pdf_path> [api_key] [query]")
        print("\nArguments:")
        print("  pdf_path  - Path to the PDF file to process")
        print("  api_key   - Google AI Studio API key (or set GOOGLE_API_KEY env var)")
        print("  query     - Question or instruction for processing the PDF")
        print("\nExamples:")
        print("  python pdf_extractor_agent.py document.pdf AIza... 'Summarize this'")
        print("  set GOOGLE_API_KEY=AIza... && python pdf_extractor_agent.py doc.pdf")
        print("\nGet your API key from: https://aistudio.google.com/apikey")
        return
    
    pdf_path = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    query = sys.argv[3] if len(sys.argv) > 3 else "Summarize this document"
    
    run_demo(pdf_path, api_key, query)


if __name__ == "__main__":
    main()
