#!/usr/bin/env python
"""
PDF to Text Extractor Agent - Main Entry Point

A simple agent that extracts text from PDF files and processes it using OpenRouter API (DeepSeek).

Usage:
    python main.py <pdf_path> [query]
    
Example:
    python main.py document.pdf "Summarize the key points"
    
Note: Set OPEN_ROUTER_API in .env file in the sampleagents folder
"""

import os
import sys
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Load environment variables from .env file
from dotenv import load_dotenv

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')

# Load .env file
load_dotenv(env_path)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(script_dir))

from sampleagents.pdf_extractor_agent import (
    create_pdf_extractor_agent_openrouter,
    create_simple_extractor_agent,
)


# Default model - DeepSeek Chat via OpenRouter
DEFAULT_MODEL = "deepseek/deepseek-chat"


def main():
    """Main entry point for the PDF Extractor Agent."""
    
    print("=" * 60)
    print("  PDF to Text Extractor Agent")
    print("  Powered by DeepSeek (via OpenRouter)")
    print("=" * 60)
    
    # Get API key from environment
    api_key = os.environ.get("OPEN_ROUTER_API")
    
    if not api_key:
        print("\n[ERROR] OPEN_ROUTER_API not found!")
        print("\nPlease create a .env file in sampleagents folder with:")
        print("  OPEN_ROUTER_API=your_api_key_here")
        print("\nGet your API key from: https://openrouter.ai/keys")
        return
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python main.py <pdf_path> [query]")
        print("\nArguments:")
        print("  pdf_path  - Path to the PDF file to process")
        print("  query     - (Optional) Question or instruction for processing")
        print("\nExamples:")
        print("  python main.py document.pdf")
        print('  python main.py report.pdf "What are the key findings?"')
        print("\nNote: API key is loaded from .env file (OPEN_ROUTER_API)")
        print(f"\nModel: {DEFAULT_MODEL}")
        return
    
    # Parse arguments
    pdf_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "Summarize this document in bullet points"
    
    # Validate PDF file exists
    if not os.path.exists(pdf_path):
        print(f"\n[ERROR] PDF file not found: {pdf_path}")
        return
    
    print(f"\n[PDF] File: {pdf_path}")
    print(f"[QUERY] {query}")
    print(f"[MODEL] {DEFAULT_MODEL}")
    print("\n" + "-" * 60)
    print("Processing...")
    
    try:
        # Create the agent with OpenRouter (DeepSeek)
        agent = create_pdf_extractor_agent_openrouter(
            api_key=api_key,
            model=DEFAULT_MODEL
        )
        
        # Run the extraction workflow
        result = agent.run_flow(
            "pdf_extraction_workflow",
            context={
                "pdf_path": pdf_path,
                "query": query
            }
        )
        
        # Display results
        print("-" * 60)
        print(f"[STATUS] {result.status.value}")
        print(f"[TIME] {result.execution_time:.2f}s")
        
        if result.success:
            report = result.task_results.get("generate_report")
            if report and report.output:
                output = report.output
                
                print("\n[DOCUMENT STATISTICS]")
                print(f"   Pages: {output.get('extraction', {}).get('total_pages', 'N/A')}")
                print(f"   Words: {output.get('extraction', {}).get('total_words', 'N/A')}")
                print(f"   Chunks: {output.get('chunks', 'N/A')}")
                
                print("\n[DEEPSEEK RESPONSE]")
                print("-" * 60)
                response = output.get('llm_response', 'No response generated')
                print(response)
                print("-" * 60)
        else:
            print(f"\n[ERROR] {result.errors}")
            
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        if "openai" in str(e).lower():
            print("\nMake sure to install dependencies:")
            print("  pip install PyMuPDF openai")


if __name__ == "__main__":
    main()
