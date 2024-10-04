import ast
import docx
import nltk
import re 
import numpy as np
import pandas as pd
import pdfplumber
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from azure.ai.formrecognizer import FormRecognizerClient
from openai import AzureOpenAI
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from azure.ai.formrecognizer import FormRecognizerClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException,UploadFile
import os
from rag_data_processing import CONNECTION_STRING, CONTAINER_NAME
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import ContentSettings
from rag_data_processing import chunks_string,generate_embeddings,upload_files_to_blob

# Initialize NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
# Azure configuration
FORM_RECOGNIZER_ENDPOINT = "https://eastus.api.cognitive.microsoft.com/"
FORM_RECOGNIZER_API_KEY = "26cfaa6c7c314e9a8ad7a68587ca3ce9"

# Initialize Azure clients
AZURE_OPENAI_API_KEY = "f619d2d04b4f44d28708e4c391039d01"
AZURE_OPENAI_ENDPOINT = "https://openainstance001.openai.azure.com/"
form_recognizer_client = FormRecognizerClient(FORM_RECOGNIZER_ENDPOINT, AzureKeyCredential(FORM_RECOGNIZER_API_KEY))
azure_openai_client = AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, api_version="2024-02-01", azure_endpoint=AZURE_OPENAI_ENDPOINT)
client = AzureOpenAI(
  api_key = "f619d2d04b4f44d28708e4c391039d01",
  api_version = "2024-02-01",
  azure_endpoint = "https://openainstance001.openai.azure.com/"

)




def extract_array_of_embedding_from_file(file_name):
    df = pd.read_csv(file_name)
    embedding_list_final = []
    embedding_list = df.embedding.apply(ast.literal_eval)
    for temp_element in embedding_list:
        embedding_list_final.append(temp_element)
    embedding_array = np.array(embedding_list_final)
    return embedding_array, df


def query_array(query, model="text-embedding-3-small"):
    data = client.embeddings.create(input = [query], model=model).data[0].embedding
    query_array = np.array(data)
    query_array = query_array.reshape(1, -1)
    return query_array




def get_url(connection_string,container_name, folder_name, blob_name):
    # # Create the BlobServiceClient object
    # blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    # Connect to Azure Blob Storage
    connection_string_blob = "DefaultEndpointsProtocol=https;AccountName=aisa0101;AccountKey=rISVuOQPHaSssHHv/dQsDSKBrywYnk6bNuXuutl4n+ILZNXx/CViS50NUn485kzsRxd5sfiVSsMi+AStga0t0g==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(f"{folder_name}/{blob_name}")
    content_settings = ContentSettings(content_type='application/pdf', content_disposition='inline')
    # Set the blob properties
    blob_client.set_http_headers(content_settings=content_settings)

    # Generate SAS token
    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
        blob_name=f"{folder_name}/{blob_name}",
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=24)  # Set expiry time as needed
    )

    # Construct the full URL to the blob
    blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{folder_name}/{blob_name}?{sas_token}"
    return blob_url



def get_text_cosine_similarity(query_array, db_array, top_k, dataframe, folder_name):
    cosine_sim = cosine_similarity(query_array, db_array)
    cosine_sim = cosine_sim.flatten()
    top_10_indices = np.argsort(cosine_sim)[-top_k:][::-1]
    top_10_df = dataframe.iloc[top_10_indices]
    print(top_10_df)
    text_list = top_10_df["text"].to_list()
    # Creating a dictionary with page_no as the key and file_name as the value
    page_file_dict = top_10_df.set_index('page_no')['file_name'].to_dict()
    # List to store the new format
    new_format_list = []

    # Fill the list with dictionaries in the new format
    for page, file in page_file_dict.items():
        file_url = get_url(CONNECTION_STRING,CONTAINER_NAME, folder_name, file)
        new_format_list.append({
            "page_numbers": int(page),
            "file_link": str(file_url),
            "file_name":str(file)
        })    
    return text_list, new_format_list


def extract_content_based_on_query(query,top_k,folder_name):
    file_name = f"{folder_name}/{folder_name}_embedding.csv"
    db_array, dataframe = extract_array_of_embedding_from_file(file_name)
    array_query = query_array(query)
    resulted_text, citation_dict = get_text_cosine_similarity(array_query, db_array, top_k, dataframe, folder_name)
    return resulted_text, citation_dict


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
def process_file(file_path, file_name, container_name, connection_string, chunk_size=200):
    # Detect file type
    file_type = detect_file_type(file_path)
    all_embeddings = []

    # Process based on file type
    if file_type == "pdf":
        pdf_type = detect_pdf_type(file_path)

        # Process text-based PDF
        if pdf_type == "text-based":
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_no, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text()
                        if text:
                            text = re.sub(r'\s+', ' ', text.strip())  # Clean up text
                            chunks = chunks_string(text, chunk_size)  # Split text into chunks
                            for chunk in chunks:
                                embedding = generate_embeddings(chunk)
                                all_embeddings.append((page_no, file_name, chunk, embedding))
            except Exception as e:
                print(f"Error processing PDF: {str(e)}")
                return []

        # Process scanned PDF
        elif pdf_type == "scanned":
            text = extract_text_from_scanned_pdf(file_path)
            if text:
                chunks = chunks_string(text, chunk_size)
                for chunk in chunks:
                    embedding = generate_embeddings(chunk)
                    all_embeddings.append((1, file_name, chunk, embedding))
            else:
                print(f"Skipping PDF '{file_name}' due to empty text extraction.")
                return []

        else:
            print(f"Skipping PDF '{file_name}' due to type detection failure.")
            return []

    # Process DOCX file
    elif file_type == "docx":
        text = extract_text_from_docx(file_path)
        if text:
            chunks = chunks_string(text, chunk_size)
            for chunk in chunks:
                embedding = generate_embeddings(chunk)
                all_embeddings.append((1, file_name, chunk, embedding))
        else:
            print(f"Skipping DOCX '{file_name}' due to empty text extraction.")
            return []

    else:
        print(f"Unsupported file type for '{file_name}'.")
        return []

    # Create DataFrame from the generated embeddings
    # Create DataFrame from the generated embeddings
    data = [t for t in all_embeddings if t]  # Remove empty tuples
    new_df = pd.DataFrame(data, columns=['page_no', 'file_name', 'text', 'embedding'])

# Save DataFrame to CSV with the correct naming format
    folder_path = os.path.dirname(file_path)
    csv_name = f"{os.path.splitext(file_name)[0]}_embedding.csv"  # Correct CSV name
    csv_path = os.path.join(folder_path, csv_name)
    new_df.to_csv(csv_path, index=False)

# Upload local files (PDF and CSV) to Azure Blob Storage
    upload_files_to_blob(folder_path, CONTAINER_NAME, CONNECTION_STRING)

# Clean up the local folder
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        print(f"Error while deleting folder {folder_path}: {e}")

    return all_embeddings
