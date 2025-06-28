import os
import requests
from PyPDF2 import PdfReader


def download_paper(pdf_url, download_dir=None):
    """
    Downloads a PDF from arXiv to the Downloads folder.

    Parameters:
    pdf_url (str): The direct URL to the PDF.
    download_dir (str, optional): Target directory. Defaults to ~/Downloads.

    Returns:
    str: Path to the downloaded file.
    """
    print("\n\n\nDownloading PDF...\n\n\n")
    if download_dir is None:
        download_dir = "./Downloads"

    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(pdf_url)
    if response.status_code == 200:
        pdf_title = pdf_url.split("/")[-1]
        pdf_title = pdf_title\
            .replace("/", "_")\
            .replace(":", "_")\
            .replace(" ", "_")
            
        file_name = pdf_title + ".pdf"
        print(file_name)
            
        file_path = os.path.join(download_dir, file_name)
        
        # Check if the paper already exists in the downloads folder
        if os.path.exists(file_path):
            print(f"Paper already exists: {file_path}")
            return file_path  # Return the existing file path
        
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded to: {file_path}")
        return file_path
    else:
        raise Exception(f"Failed to download PDF. HTTP Status: {response.status_code}")
    
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def summarize(text, llm):
    response = llm.invoke(
        input=[
            {
            "role": "system",
            "content": "You are PDF summariser. You take text extracted from PDF and summarise it.",
            },
            {
            "role": "user",
            "content": text,
            },
        ],
    )
    return response.content