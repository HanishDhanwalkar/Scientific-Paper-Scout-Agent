# server.py
from mcp.server.fastmcp import FastMCP
import arxiv
import os

from helper import summarize, extract_text_from_pdf, download_paper

from llm_handler import get_summarizer_llm_client
from config import DOWNLOADS_DIR, CURRENT_SUMMARIZER_LLM_PROVIDER, CURRENT_LLM_PROVIDER

llm = get_summarizer_llm_client()
print(f"Using summarizer LLM: {CURRENT_SUMMARIZER_LLM_PROVIDER}")
print(f"Using chatbot LLM: {CURRENT_LLM_PROVIDER}")

print("Staring Server....")
mcp = FastMCP(name="Demo") # Create an MCP server


# # Add an addition tool
# @mcp.tool()
# def add(a: int, b: int) -> int:
#     """Add two numbers"""
#     return a + b

# # Add a resource
# @mcp.resource("greeting://{name}")
# def get_greeting(name: str) -> str:
#     """Get a personalized greeting"""
#     return f"Hello, {name}!"

@mcp.tool()
def query_arxiv(query: str, max_results: int=3) -> list[dict]:
    """
    Queries the public arXiv API using the 'arxiv' Python package
    and returns up to max_results items.

    Args:
        query (str): The search query string (e.g., "large language models").
        max_results (int): The maximum number of results to return.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents an
                    arXiv article with its title, authors, summary, and primary link.
                    Returns an empty list if no results are found or an error occurs.
    """
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for result in search.results():
            results.append(
                {
                "title": result.title,
                "pdf_url": result.pdf_url,
                # "summary": result.summary
                }
            )
        return results

    except Exception as e:
        print(f"An error occurred while querying arXiv: {e}")
        return []

@mcp.tool()
def summarize_paper(pdf_url):
    """
    Summarize a paper from pdf_url of the paper

    Args:
        pdf_url (str): The URL of the PDF to summarize

    Returns:
        str: A summary of the PDF
    """
    global DOWNLOADS_DIR
    
    if pdf_url is None:
        raise ValueError("PDF URL is required.")
    
    pdf_path = download_paper(pdf_url=pdf_url, download_dir=DOWNLOADS_DIR)
    
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"{pdf_path} does not exist.")
    
    text = extract_text_from_pdf(pdf_path)

    # limiting very large texts
    if len(text) > 15000:
        print("PDF is too long, summarizing only the first 8000 characters.")
        text = text[:8000]
        
    global llm

    print("Sending to Ollama for summarization...")
    summary = summarize(text, llm)
    return summary