import arxiv
from  helper import download_paper

def search_paper(input_data):
    """
    Searches arXiv for papers using a query string.

    Parameters:
    input_data (dict): A dictionary with keys:
        - "query" (str): Search term.
        - "max_results" (int): Max number of results to return.

    Returns:
    list of dicts: Each dict contains 'title', 'pdf_url', and 'summary'.
    """
    query = input_data.get("query")
    max_results = input_data.get("max_results", 5)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    for result in search.results():
        results.append({
            "title": result.title,
            "pdf_url": result.pdf_url,
            "summary": result.summary
        })
    return results


if __name__ == "__main__":
    # Search for papers
    papers = search_paper({"query": "transformers in NLP", "max_results": 3})

    # Show and download
    for paper in papers:
        print(paper["title"])
        print(paper["pdf_url"])
        download_paper(paper["pdf_url"])

    ############## OUTPUT ###################
    # GR-NLP-TOOLKIT: An Open-Source NLP Toolkit for Modern Greek
    # http://arxiv.org/pdf/2412.08520v1
    # GR-NLP-TOOLKIT__An_Open-Source_NLP_Toolkit_for_Modern_Greek.pdf
    # Paper already exists: ./Downloads\GR-NLP-TOOLKIT__An_Open-Source_NLP_Toolkit_for_Modern_Greek.pdf
    # On the Equivalence between Logic Programming and SETAF
    # http://arxiv.org/pdf/2407.05538v1
    # On_the_Equivalence_between_Logic_Programming_and_SETAF.pdf
    # Paper already exists: ./Downloads\On_the_Equivalence_between_Logic_Programming_and_SETAF.pdf
    # Noisy Text Data: Achilles' Heel of popular transformer based NLP models
    # http://arxiv.org/pdf/2110.03353v1
    # Noisy_Text_Data__Achilles'_Heel_of_popular_transformer_based_NLP_models.pdf
    # Paper already exists: ./Downloads\Noisy_Text_Data__Achilles'_Heel_of_popular_transformer_based_NLP_models.pdf