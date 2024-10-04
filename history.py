# Function to store data in a JSON file
import json 
import os 


def store_data(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


# Function to get data from a JSON file
def get_data(file_path):
    if not os.path.exists(file_path):
        return {"chat": []}  # Return an empty structure if the file does not exist
    with open(file_path, 'r') as file:
        return json.load(file)


# Function to append data to a JSON file
def append_data(file_path, new_data):
    # Load existing data
    data = get_data(file_path)
    print("a")
    # Append the new data
    data['chat'].append(new_data)
    print("ab")
    # Store the updated data
    store_data(file_path, data)



import mysql.connector
from mysql.connector import Error
import json


# Connection details
host_name = "mysqlai.mysql.database.azure.com"
user_name = "azureadmin"
user_password = "Meridian@123"
database_name = "chatbot"




def store_data_in_db(data, user_id, db_name):
    try:
        # Establish the connection
        connection = mysql.connector.connect(
            host=host_name,
            database=database_name,
            user=user_name,
            password=user_password
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Define the insert query
            insert_query = """
            INSERT INTO km_chat_history (user_id, db_name, user_question, bot_answer, citations)
            VALUES (%s, %s, %s, %s, %s)
            """

            # Extract and insert data
            print("data",data)
            for item in data:
                user_question = item.get("user_question")
                bot_answer = item.get("answer", {}).get("bot_answer")
                citations = item.get("answer", {}).get("citation_dict", [])

                # Serialize citations as a JSON string
                citations_json = json.dumps(citations)
                print(user_id, db_name, user_question, bot_answer, citations_json)
                cursor.execute(insert_query, (user_id, db_name, user_question, bot_answer, citations_json))
            
            connection.commit()
            print("Data inserted successfully into the database.")

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")


def extract_and_format_data(db_name, user_id):
    try:
        # Establish the connection
        connection = mysql.connector.connect(
            host=host_name,
            database=database_name,
            user=user_name,
            password=user_password
        )

        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            # Define the select query with filters
            select_query = """
            SELECT user_question, bot_answer, citations 
            FROM km_chat_history
            WHERE db_name = %s AND user_id = %s
            """

            # Execute the query with parameters
            cursor.execute(select_query, (db_name, user_id))
            rows = cursor.fetchall()

            # Format the data
            formatted_data = []

            for row in rows:
                user_question = row["user_question"]
                bot_answer = row["bot_answer"]
                citations = json.loads(row["citations"])  # Deserialize JSON string
                
                formatted_data.append({
                    "user_question": user_question,
                    "answer": {
                        "bot_answer": bot_answer,
                        "citation_dict": citations
                    }
                })

            return formatted_data

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")


def count_rows_and_databases_by_user(user_id):
    try:
        # Establish the connection
        connection = mysql.connector.connect(
            host=host_name,
            database=database_name,
            user=user_name,
            password=user_password
        )
        if connection.is_connected():
            cursor = connection.cursor()

            # Define the query to count rows and distinct databases
            count_query = """
            SELECT COUNT(*) AS row_count, COUNT(DISTINCT db_name) AS db_count
            FROM km_chat_history
            WHERE user_id = %s
            """

            # Execute the query with the user_id parameter
            cursor.execute(count_query, (user_id,))
            result = cursor.fetchone()

            row_count = result[0]  # Total number of rows for the user_id
            db_count = result[1]   # Number of distinct databases for the user_id

            return {"conversation_count": row_count, "database_count": db_count}

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")         
           
  