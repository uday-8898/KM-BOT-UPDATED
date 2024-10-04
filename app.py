from openai import AzureOpenAI
from datetime import datetime
from model import extract_content_based_on_query
import json
from fastapi import Query
import re
from typing import Dict
from azure.core.exceptions import HttpResponseError

from rag_data_processing import extact_content_embedding_from_file, read_and_split_pdf,upload_files_to_blob
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import os
from rag_data_processing import CONNECTION_STRING, CONTAINER_NAME
from pydantic import BaseModel
import time 
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
import shutil
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from history import *
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
from mysql.connector import Error
import random
import string
from model import  extract_content_based_on_query,process_file




# Connection details
host_name = "mysqlai.mysql.database.azure.com"
user_name = "azureadmin"
user_password = "Meridian@123"
db_name = "chatbot"


db_config = {
    "host": host_name,
    "database": db_name,
    "user": user_name,
    "password": user_password
}


chat_client = AzureOpenAI(
  azure_endpoint = "https://openainstance001.openai.azure.com/", 
  api_key="f619d2d04b4f44d28708e4c391039d01",  
  api_version="2024-03-01-preview"
)


def get_response_from_query(query, content, history, language):
    message = [
        {"role": "system", "content": f"You are an AI assistant that helps to answer the questions from the given content in {language} language. Give the response in JSON."},
        {"role": "user", "content": f"""Your task is to follow chain of thought method to first extract accurate answer for given user query, chat history and provided input content. Then change the language of response into {language} language. Give the response in the json format only having 'bot answer' and 'scope' as key.\n\nInput Content : {content} \n\nUser Query : {query}\n\nChat History : {history}\n\nImportant Points while generating response:\n1. The answer of the question should be relevant to the input text.\n2. Answer complexity would be based on input content.\n3. If input content is not provided direct the user to provide content.\n4. Answers should not be harmful or spam. If there is such content give the instructions to user accordingly. \n5. If user query is out of scope of given content give the value of 'scope' key False.\n6. Give the response in the json format. \n\nExtracted json response:"""}
    ]

    response = chat_client.chat.completions.create(
      model="gpt4", # model = "deployment_name"
      messages = message,
      temperature=0.7,
      max_tokens=500,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None,
      response_format = {"type": "json_object"}
    )
    # Loading the response as a JSON object
    json_response = json.loads(response.choices[0].message.content)
    print(json_response)
    return json_response


def language_correct_query(query, history):
    message = [
        {"role": "system", "content": "You are an AI assistant that helps to identify and extract the language, fixes the typing error and change the any language into english language content by understanding the user query."},
        {"role": "user", "content": f"""Your task is to helps to identify and extract the language of query string, fixes the typing error and change the any language into english language content. Give the response always in the json format only. \n\nInput Content : {query} \n\nHistory : {history}\n\nImportant instructions: \n1. Your task is to identify the language of content.(e.g. : english/french/..)\n2. You have to generate the modified content by fixing the typing error and change the language of input content into english language if it is other than english language content.\n\nKey Entities for the json response: \n1. Language\n2. Modified Content\n\nExtracted Json Response :"""}
    ]

    response = chat_client.chat.completions.create(
      model="gpt-4o-mini", # model = "deployment_name"
      messages = message,
      temperature=0.7,
      max_tokens=300,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None,
      response_format = {"type": "json_object"}

    )
    # Loading the response as a JSON object
    json_response = json.loads(response.choices[0].message.content)
    return json_response

def generate_random_string(length=10):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string



# Define the query request model
class QueryRequest(BaseModel):
    query : str
    database : str
    email : str

class DownloadRequest(BaseModel):
    folder_name: str

def background_task(folder_path: str):
    # Simulate a long-running task
    _ = extact_content_embedding_from_file(folder_path)
    print(f"Background task completed ")

# Define the response model
class QueryResponse(BaseModel):
    bot_answer: str
    citation_dict: list


# Pydantic model for request validation
class UserRegistration(BaseModel):
    name: str
    email: str


def download_blobs_from_folder(container_name, folder_name, connection_string, local_download_path):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    folder_path = os.path.join(local_download_path, folder_name)
    
    # Create local download path if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    blob_list = container_client.list_blobs(name_starts_with=folder_name)
    csv_blobs = [blob for blob in blob_list if blob.name.endswith('.csv')]
    
    if not csv_blobs:
        print("No .csv files found in the folder.")
        return False

    for blob in csv_blobs:
        blob_client = container_client.get_blob_client(blob.name)
        local_file_path = os.path.join(folder_path, os.path.relpath(blob.name, folder_name))
        
        # Create directories if they don't exist
        local_dir = os.path.dirname(local_file_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        
        with open(local_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        print(f"Downloaded {blob.name} to {local_file_path}")
    
    return True


def respond_to_question(original_query_string, folder_name, email_id):
    # if csv not present
    current_working_directory = os.getcwd()
    db_path = os.path.join(current_working_directory, folder_name)
    if not os.path.exists(db_path):
        result = download_blobs_from_folder(CONTAINER_NAME, folder_name, CONNECTION_STRING, current_working_directory)
        if result == False:
            return {"bot_answer": "Data Base not craeted yet", "citation_dict": {}}
    history = " "     
    # This function should already exist with the required logic
    language_response = language_correct_query(original_query_string, history)
    # Placeholder response logic
    query_string = language_response["Modified Content"] 
    content_list, citation_dict = extract_content_based_on_query(query_string, 10,folder_name)
    content = " ".join(content_list)
    answer = get_response_from_query(query_string, content, history, language_response["Language"].strip().lower())
    if answer["scope"] == False:
        citation_dict = []
    output_response = {"bot_answer": answer["bot answer"], "citation_dict": citation_dict}  
    db_response = [{"user_question": original_query_string, "answer" : output_response }]
    user_id = get_user_id_by_email(email_id)    
    store_data_in_db(db_response, user_id, folder_name)
    # append_data(history_data_path, output_response)
    return output_response



def get_user_id_by_email(email_id):
    # Database connection configuration
    db_config = {
    "host": host_name,
    "database": db_name,
    "user": user_name,
    "password": user_password
}

    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)

        if connection.is_connected():
            cursor = connection.cursor()
            # Query to fetch user_id based on email_id
            fetch_user_id_query = """
            SELECT user_id FROM km_registration WHERE email_id = %s
            """
            cursor.execute(fetch_user_id_query, (email_id,))
            result = cursor.fetchone()

            if result:
                user_id = result[0]
                return user_id
            else:
                return None

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def add_db_mapping(user_id, database_name):
    # Database connection configuration
    db_config = {
    "host": host_name,
    "database": db_name,
    "user": user_name,
    "password": user_password
}


    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)

        if connection.is_connected():
            cursor = connection.cursor()
            # Insert the user_id and db_name into the km_db_mapping table
            insert_query = """
            INSERT INTO km_db_mapping (user_id, db_name) VALUES (%s, %s)
            """
            cursor.execute(insert_query, (user_id, database_name))
            connection.commit()
            return {"message": "Mapping added successfully", "user_id": user_id, "db_name": database_name}

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return {"error": str(e)}

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()



def get_db_names_by_user_id(user_id):
    # Database connection configuration
    db_config = {
    "host": host_name,
    "database": db_name,
    "user": user_name,
    "password": user_password
}


    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)

        if connection.is_connected():
            cursor = connection.cursor()
            # Query to fetch db_names for the given user_id
            fetch_db_names_query = """
            SELECT db_name FROM km_db_mapping WHERE user_id = %s
            """
            cursor.execute(fetch_db_names_query, (user_id,))
            results = cursor.fetchall()

            # Extract db_names from the results
            db_names = [row[0] for row in results]
            
            return {"user_id": user_id, "db_names": db_names}

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return {"error": str(e)}

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


# Pydantic model for request validation
class EmailRequest(BaseModel):
    email: str

# Define the input model
class History(BaseModel):
    user_id: int
    database: str


# Define the input model
class Count(BaseModel):
    user_id: int


app = FastAPI()


origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/chat/history", response_model=List[dict])
async def get_commands(payload: History):

    response = extract_and_format_data(  payload.database, payload.user_id)
    return response



@app.post("/user/summary", response_model=dict)
async def get_commands(count: Count):
    response = count_rows_and_databases_by_user(count.user_id)
    return response



@app.post("/auth/login", response_model=dict)
async def register_user(user: UserRegistration):
    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)

        if connection.is_connected():
            cursor = connection.cursor()
            
            # Check if the email already exists in the database and get the user_id
            email_check_query = "SELECT user_id FROM km_registration WHERE email_id = %s"
            cursor.execute(email_check_query, (user.email,))
            result = cursor.fetchone()
            
            if result:
                user_id = result[0]  # Extract the user_id from the result
                return {"message": "Email already registered", "user_id": user_id}
            
            # Generate a 10-digit random string for the password (or use user.password)
            random_string = generate_random_string(10)
            
            # Insert user details into the database
            insert_query = """
            INSERT INTO km_registration (name, email_id, password) VALUES (%s, %s, %s)
            """
            cursor.execute(insert_query, (user.name, user.email, random_string))
            connection.commit()
            user_id = cursor.lastrowid  # Get the last inserted user_id

            return {"message": "User registered successfully", "user_id": user_id}

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()




# Endpoint for user registration
@app.post("/auth/login", response_model=dict)
async def register_user(user: UserRegistration):
    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)

        if connection.is_connected():
            cursor = connection.cursor()
                        # Generate a 10-digit random string
            random_string = generate_random_string(10)
            # Insert user details into the database
            insert_query = """
            INSERT INTO km_registration (name, email_id, password) VALUES (%s, %s , %s)
            """
            cursor.execute(insert_query, (user.name, user.email, random_string))
            connection.commit()
            user_id = cursor.lastrowid  # Get the last inserted user_id

            return {"message": "User registered successfully", "user_id": user_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()




@app.post("/databases")
def list_folders(request: EmailRequest):
    try:
        email_str = request.email
        user_id = get_user_id_by_email(email_str)
        if user_id == None:
            return {"databases": []}
        print(user_id)
        db_names = get_db_names_by_user_id(user_id)
        db_names = db_names["db_names"]
        print(db_names)
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blobs = container_client.walk_blobs()
        
        # Extract folder names (prefixes)
        folders = set()
        for blob in blobs:
            folder_path = os.path.dirname(blob.name)
            if folder_path:  # Only add if it's not an empty string
                folders.add(folder_path)
        folder_list = list(folders)
        final_db_list = []
        for db in db_names:
            if db in folder_list:
                final_db_list.append(db)
        return {"databases": final_db_list}
    except ResourceNotFoundError:
        raise HTTPException(status_code=404, detail="Container not found")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        response = respond_to_question(request.query, request.database, request.email)
        return QueryResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/database/create")
async def trigger_task(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),  # Uploaded files
    folder_name: str = Form(...),  # Folder to save files
    email: str = Form(...),  # User email
    file_type: str = Form(...)  # Type of data: 'scanned_pdf', 'standard_pdf', 'docx', etc.
):
    try:
        user_id = get_user_id_by_email(email)
        _ = add_db_mapping(user_id, folder_name)
        # Create folder if it doesn't exist
        folder_path = os.path.join(os.getcwd(), folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Save files to the folder
        for file in files:
            file_path = os.path.join(folder_path, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        if file_type == 'pdf':
            files = os.listdir(folder_name)
            pdf_files = [f for f in files] 
            total_chunks = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(folder_path, pdf_file)
                print(f"Reading {pdf_file}...")
                chunks = read_and_split_pdf(pdf_path, pdf_file)
                total_chunks += chunks  # Accumulate total chunks  
            if len(total_chunks) <= 0: 
                minutes_to_wait = 0    
            minutes_to_wait = (len(total_chunks) * 2)/60  
            minutes_to_wait = round(minutes_to_wait, 2)
            # Add the background task
            background_tasks.add_task(background_task, folder_name)
            response = {
        "message": "Database created successfully",
        "eta": f"{minutes_to_wait} minutes"
        }

            return response
        if file_type == 'scanned_pdf':
            # Special logic for scanned PDFs
            for file in files:
                file_location = os.path.join(folder_path, file.filename)
                # Logic to process scanned PDFs
                        # Process the file and generate embeddings
                process_file(file_location,file.filename,CONTAINER_NAME, CONNECTION_STRING)
        # print(embeddings)

        # Upload the file to Azure Blob Storage
        # upload_files_to_blob(file_location, CONTAINER_NAME, CONNECTION_STRING)

        # Return the response with embeddings
            return JSONResponse(content={"message": "File processed successfully"})

    
    except Exception as Argument:
        print(Argument)
        # creating/opening a file
        f = open("log.txt", "a")
        # writing in the file
        f.write(str(Argument))
        # closing the file
        f.close() 
        response = {
        "message": "Database not created",
        "eta": f"0 minutes"
        }

        return response

# # Endpoint to upload and process PDF/DOCX
# @app.post("/upload_file/")
# async def upload_file(file: UploadFile = File(...)):
#     try:
#         # Create a folder named after the file (without extension) and save the file inside it
#         file_name_without_extension = os.path.splitext(file.filename)[0]
#         local_folder_path = os.path.join(file_name_without_extension)
#         os.makedirs(local_folder_path, exist_ok=True)
        
#         file_location = os.path.join(local_folder_path, file.filename)
#         with open(file_location, "wb+") as file_object:
#             file_object.write(file.file.read())

#         # Process the file and generate embeddings
#         process_file(file_location,file.filename,CONTAINER_NAME, CONNECTION_STRING)
#         # print(embeddings)

#         # Upload the file to Azure Blob Storage
#         # upload_files_to_blob(file_location, CONTAINER_NAME, CONNECTION_STRING)

#         # Return the response with embeddings
#         return JSONResponse(content={"message": "File processed successfully"})

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=7000)


















