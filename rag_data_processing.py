import zipfile
import os
import re
import sys
import shutil
import itertools
import requests
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from openai import AzureOpenAI
import nltk
from nltk.tokenize import sent_tokenize
from azure.storage.blob import BlobServiceClient




client = AzureOpenAI(
  api_key = "f619d2d04b4f44d28708e4c391039d01",
  api_version = "2024-02-01",
  azure_endpoint = "https://openainstance001.openai.azure.com/"
)


# Split Content into chunks
def chunks_string(text, tokens):
    # Initialize variables
    segments = []
    len_sum = 0
    k = 0

    # Split the text into sentences
    raw_list = sent_tokenize(text)

    # Iterate the Sentences one-by-one
    for i in range(len(raw_list)):

      # Split that sentence into tokens
      x1 = len(raw_list[i].split())

      # Cummulative length of tokens till this sentence
      len_sum = len_sum + x1

      k = k + 1

      # If no. of tokens > threshold
      if len_sum > tokens:

        ### Logic for finding how many sentences need to be repeat in current segment ###

        # Will be used for first segment only
        if i-(k+1) < 0:
            j = 0

        # Will be used for next  all segments
        else:
          j = i-(k+1)
          if len(" ".join(raw_list[j: i+1]).split()) > tokens:
            j = i-k

        # Append list of sentences to each segment
        segments.append(" ".join(raw_list[j: i]))

        # Set variables = 0
        len_sum = 0
        k = 0

      # If it is last iteration
      if i == len(raw_list)-1:
        if i-(k+1) < 0:
          j = 0

        else:
          j = i-(k+1)
          if len(" ".join(raw_list[j: i+1]).split()) > tokens:
            j = i-k

          # Append list of sentences to each segment
          segments.append(" ".join(raw_list[j: i+1]))

    return segments



# Function to read PDF file content and split into chunks
def read_and_split_pdf(file_path, file_name, chunk_size=200):
    reader = PdfReader(file_path)
    content_chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_content = page.extract_text() or ''
        # Split content into chunks based on word count
        chunks = chunks_string(page_content, chunk_size)
        content_chunks.extend([(page_num,file_name, chunk.strip()) for chunk in chunks if len(chunk.split()) > 2])
    return content_chunks


# print (generate_embeddings(text_chunks[1]))
def generate_embeddings(texts, model="text-embedding-3-small"):
    return client.embeddings.create(input=[texts], model=model).data[0].embedding



# Set your Azure Blob Storage details
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=aisa0101;AccountKey=rISVuOQPHaSssHHv/dQsDSKBrywYnk6bNuXuutl4n+ILZNXx/CViS50NUn485kzsRxd5sfiVSsMi+AStga0t0g==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "aibot"
# LOCAL_FOLDER_PATH = "folder1"  # Set your local folder path here


# Function to upload local PDF files to Azure Blob Storage, preserving folder name
def upload_files_to_blob(local_folder_path, container_name, connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    local_folder_name = os.path.basename(os.path.normpath(local_folder_path))  # Get the local folder name

    for root, _, files in os.walk(local_folder_path):
        for file in files:
            # if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, local_folder_path)  # Get relative path
                blob_name = f"{local_folder_name}/{relative_path}".replace('\\', '/')  # Include folder name in blob path
                blob_client = container_client.get_blob_client(blob_name)
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True, connection_timeout=600)
                print(f"Uploaded {file} to {container_name}/{blob_name}")


def upload_files_to_blob(local_folder_path, container_name, connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    local_folder_name = os.path.basename(os.path.normpath(local_folder_path))  # Get the local folder name

    for root, _, files in os.walk(local_folder_path):
        for file in files:
            # if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, local_folder_path)  # Get relative path
                blob_name = f"{local_folder_name}/{relative_path}".replace('\\', '/')  # Include folder name in blob path
                blob_client = container_client.get_blob_client(blob_name)
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True, connection_timeout=600)
                print(f"Uploaded {file} to {container_name}/{blob_name}")
# Function to upload local PDF files to Azure Blob Storage, preserving folder name


def extact_content_embedding_from_file(folder_path):
    # # Path to the folder containing PDF files
    # folder_path = '/content/Ncert'
    db_path = f"{folder_path}/{folder_path}_embedding.csv"
    old_files_list = []
    # List all files in the folder
    files = os.listdir(folder_path)
    old_df = pd.DataFrame(columns=['page_no', 'file_name', 'text',"embedding"])

    # Filter out PDF files
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]    


    if os.path.exists(db_path):
        old_df = pd.read_csv(f"{folder_path}/{folder_path}_embedding.csv")
        old_files_list =  old_df['file_name'].unique()
        # remove rows from old df which is no longer in folder
        for old_file in old_files_list:
            if old_file not in pdf_files:
                print("deleting file ", old_file)
                # Condition to remove rows where 'Column1' is greater than 3
                condition = old_df['file_name'] == old_file
                # Remove rows based on condition
                old_df = old_df[~condition]

    # Total number of chunks
    total_chunks = []
    embedding_list = []

    # Read each PDF file, split into chunks, and display page number and chapter name
    for pdf_file in pdf_files:
        if pdf_file not in old_files_list:
            pdf_path = os.path.join(folder_path, pdf_file)
            print(f"Reading {pdf_file}...")
            chunks = read_and_split_pdf(pdf_path, pdf_file)
            total_chunks += chunks  # Accumulate total chunks
            print("Number of chunks:", len(chunks))
            print("Chunks:")
            for page_num, file_name, chunk in total_chunks:
                print(f"Page {page_num} : Filename {file_name}: {chunk}")

    print("Total number of chunks from all PDF files:", total_chunks)
    for i, chunk in enumerate(total_chunks):

        embedding = generate_embeddings(chunk[2])
        embedding_list.append(embedding)

    # Remove empty tuples
    data = [t for t in total_chunks if t]

    # Create DataFrame
    new_df = pd.DataFrame(data, columns=['page_no', 'file_name', 'text'])
    new_df['embedding'] = embedding_list


    new_df = pd.concat([ new_df, old_df], ignore_index=True)
    new_df.to_csv(f"{folder_path}/{folder_path}_embedding.csv", index = False)

    # Upload local PDF files to Azure Blob Storage
    upload_files_to_blob(folder_path, CONTAINER_NAME, CONNECTION_STRING)
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        print(e)    
    return True





# extact_content_embedding_from_file(r'C:\Users\RoshanKumar\OneDrive - Meridian Solutions\Desktop\ChatbotAPI\Ncert')
# df = pd.read_csv("embedding.csv")
# print(df.file_name.unique())