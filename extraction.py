import os
import re
import pdfplumber
import docx
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from azure.storage.blob import BlobServiceClient
from azure.ai.formrecognizer import FormRecognizerClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from openai import AzureOpenAI
from nltk.tokenize import sent_tokenize
from typing import List
 
app = FastAPI()
 
# Azure configuration
FORM_RECOGNIZER_ENDPOINT = "https://eastus.api.cognitive.microsoft.com/"
FORM_RECOGNIZER_API_KEY = "26cfaa6c7c314e9a8ad7a68587ca3ce9"
AZURE_OPENAI_API_KEY = "f619d2d04b4f44d28708e4c391039d01"
AZURE_OPENAI_ENDPOINT = "https://openainstance001.openai.azure.com/"
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=aisa0101;AccountKey=rISVuOQPHaSssHHv/dQsDSKBrywYnk6bNuXuutl4n+ILZNXx/CViS50NUn485kzsRxd5sfiVSsMi+AStga0t0g==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "aibot"
 
# Initialize Azure clients
form_recognizer_client = FormRecognizerClient(FORM_RECOGNIZER_ENDPOINT, AzureKeyCredential(FORM_RECOGNIZER_API_KEY))
azure_openai_client = AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, api_version="2024-02-01", azure_endpoint=AZURE_OPENAI_ENDPOINT)
 
# Detect file type (PDF or DOCX)
def detect_file_type(file_path):
    if file_path.endswith(".pdf"):
        return "pdf"
    elif file_path.endswith(".docx"):
        return "docx"
    return None
 
# Detect PDF type (text-based or scanned)
def detect_pdf_type(pdf_file_path):
    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and len(text.strip()) > 10:
                    return "text-based"
        return "scanned"
    except Exception as e:
        print(f"Error detecting PDF type: {str(e)}")
        return None
 
# Extract text from text-based PDFs
def extract_text_from_pdf(pdf_file_path):
    try:
        text = ""
        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''
        return re.sub(r'\s+', ' ', text.strip())
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None
 
# Extract text from DOCX files
def extract_text_from_docx(docx_file_path):
    try:
        doc = docx.Document(docx_file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return re.sub(r'\s+', ' ', ' '.join(full_text).strip())
    except Exception as e:
        print(f"Error extracting text from DOCX: {str(e)}")
        return None
 
# Extract text from scanned PDFs using Azure Form Recognizer
def extract_text_from_scanned_pdf(pdf_file_path):
    try:
        with open(pdf_file_path, "rb") as f:
            pdf_bytes = f.read()
 
        poller = form_recognizer_client.begin_recognize_content(pdf_bytes)
        result = poller.result()
 
        extracted_text = ""
        for page in result:
            for line in page.lines:
                extracted_text += line.text.strip() + " "
 
        return extracted_text.strip()
    except HttpResponseError as e:
        print(f"Azure Form Recognizer error: {e.message}")
        return None
    except Exception as e:
        print(f"Unexpected error with Form Recognizer: {str(e)}")
        return None
 
# Split text into chunks
def chunks_string(text, tokens):
    segments = []
    len_sum = 0
    k = 0
    raw_list = sent_tokenize(text)
 
    for i in range(len(raw_list)):
        x1 = len(raw_list[i].split())
        len_sum += x1
        k += 1
 
        if len_sum > tokens:
            j = max(0, i - k)
            segments.append(" ".join(raw_list[j:i]))
            len_sum = 0
            k = 0
 
        if i == len(raw_list) - 1:
            j = max(0, i - k)
            segments.append(" ".join(raw_list[j:i + 1]))
 
    return segments
 
# Generate embeddings for text chunks
def generate_embeddings(texts, model="text-embedding"):
    return azure_openai_client.embeddings.create(input=[texts], model=model).data[0].embedding
 
# Upload file to Azure Blob Storage
def upload_file_to_blob(file_path):
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
 
    file_name = os.path.basename(file_path)
    blob_client = container_client.get_blob_client(file_name)
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"Uploaded {file_name} to Azure Blob Storage.")
 
 
# Save embeddings to CSV file
def save_embeddings_to_csv(embeddings, filename='embeddings.csv'):
    df = pd.DataFrame(embeddings, columns=["file_name", "page_no", "chunk", "embedding"])
    df.to_csv(filename, index=False)
    print(f"Embeddings saved to {filename}.")
 
# Upload embeddings CSV to Azure Blob Storage
def upload_embeddings_to_blob(file_path):
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
 
    blob_client = container_client.get_blob_client(os.path.basename(file_path))
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"Uploaded {file_path} to Azure Blob Storage.")
 
 
 
 
# Process PDF or DOCX and generate embeddings
def process_file(file_path, file_name, chunk_size=200):
    file_type = detect_file_type(file_path)
    all_embeddings = []
 
    if file_type == "pdf":
        pdf_type = detect_pdf_type(file_path)
        if pdf_type == "text-based":
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_no, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text()
                        if text:
                            text = re.sub(r'\s+', ' ', text.strip())
                            chunks = chunks_string(text, chunk_size)
                            for chunk in chunks:
                                embedding = generate_embeddings(chunk)
                                all_embeddings.append((file_name, page_no, chunk, embedding))
            except Exception as e:
                print(f"Error processing PDF: {str(e)}")
                return []
        elif pdf_type == "scanned":
            text = extract_text_from_scanned_pdf(file_path)
            if text:
                chunks = chunks_string(text, chunk_size)
                for chunk in chunks:
                    embedding = generate_embeddings(chunk)
                    all_embeddings.append((file_name, 1, chunk, embedding))
            else:
                print(f"Skipping PDF '{file_name}' due to empty text extraction.")
                return []
        else:
            print(f"Skipping PDF '{file_name}' due to type detection failure.")
            return []
    elif file_type == "docx":
        text = extract_text_from_docx(file_path)
        if text:
            chunks = chunks_string(text, chunk_size)
            for chunk in chunks:
                embedding = generate_embeddings(chunk)
                all_embeddings.append((file_name, 1, chunk, embedding))
        else:
            print(f"Skipping DOCX '{file_name}' due to empty text extraction.")
            return []
    else:
        print(f"Unsupported file type for '{file_name}'.")
        return []
 
    # Save embeddings to CSV
    save_embeddings_to_csv(all_embeddings)
 
    # Upload the CSV to Azure Blob Storage
    upload_embeddings_to_blob('embeddings.csv')
 
    return all_embeddings
 
# Endpoint to upload and process PDF/DOCX
@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save uploaded file to local storage
        file_location = f"uploads/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
 
        # Process the file and generate embeddings
        file_name = file.filename
        embeddings = process_file(file_location, file_name)
 
        # Upload the file to Azure Blob Storage
        upload_file_to_blob(file_location)
 
        # Return the response with embeddings
        return JSONResponse(content={"message": "File processed successfully", "embeddings": embeddings})
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
 
# Start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8007)