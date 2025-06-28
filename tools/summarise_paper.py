# import os
# from PyPDF2 import PdfReader

# from langchain_ollama import ChatOllama



# DOWNLOADS_DIR = "./Downloads"
# SUMMARISER_MODEL="llama3.2"
# llm = ChatOllama(model=SUMMARISER_MODEL)


# def extract_text_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text() or ""
#     return text

# def summarize_with_ollama(text, llm):
#     response = llm.invoke(
#         input=[
#             {
#             "role": "system",
#             "content": "You are PDF summariser. You take text extracted from PDF and summarise it.",
#             },
#             {
#             "role": "user",
#             "content": text,
#             },
#         ],
#     )
#     return response.content
    

# def summarize_pdf_from_downloads(pdf_filename):
#     pdf_path = os.path.join(DOWNLOADS_DIR, pdf_filename)
#     if not os.path.isfile(pdf_path):
#         raise FileNotFoundError(f"{pdf_path} does not exist.")
    
#     print(f"Extracting text from: {pdf_path}")
#     text = extract_text_from_pdf(pdf_path)

#     # Optional: limit very large texts
#     if len(text) > 8000:
#         print("PDF is too long, summarizing only the first 8000 characters.")
#         text = text[:8000]
        
#     global llm

#     print("Sending to Ollama for summarization...")
#     summary = summarize_with_ollama(text, llm)
#     return summary


# if __name__ == "__main__":    
#     pdf_filename = "paper.pdf"
#     summary = summarize_pdf_from_downloads(pdf_filename)
#     print("Summary:")
#     print(summary)