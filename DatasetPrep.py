import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('/Users/marvinzacarias/Downloads/IMDB-Movie-Data.csv')

# Get the count of rows
row_count = len(df)
# Print the count
print(f"Number of rows: {row_count}")

# Display the first few rows of the dataframe
df.head()

from pymongo import MongoClient
import requests
import concurrent.futures
import pandas as pd
from urllib.parse import quote_plus
import json

with open('config.json', 'r') as file:
    config = json.load(file)
mongo_connection_string = config["MONGO_CONNECTION_STRING"]
client = MongoClient(mongo_connection_string)
api_key = config["API_KEY"]

def fetch_movie_data(title):
    encoded_title = quote_plus(title)
    url = f"http://www.omdbapi.com/?t={encoded_title}&apikey={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('Response') == 'True':
            return data
        else:
            print(f"No data available for {title}")
    else:
        print(f"Error {response.status_code} fetching data for {title}")
    
    return None

# Connect to your MongoDB instance
client = MongoClient('mongodb+srv://jacobrz030:oOnUj5jifqgokKZI@movies.2qhakzv.mongodb.net/?retryWrites=true&w=majority')

# Connect to your database
db = client['IMDB']

# Get your collection
collection = db['Movies']

# Get the list of unique movie titles from your DataFrame
movies = df['Title'].str.lower().str.strip().tolist()

# Use a ThreadPool to fetch data in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    movie_data = list(executor.map(fetch_movie_data, movies))

# Filter out None values
movie_data = [data for data in movie_data if data is not None]

# Convert fetched movie data into a dataframe
df1 = pd.DataFrame(movie_data)

# Preprocess data
df1 = df1.fillna('')  # Fill NaN values with an empty string

df1['Plot'] = df1['Plot'].str.lower().str.replace('[^\w\s]', '', regex=True)
df1['Actors'] = df1['Actors'].str.lower().str.replace('[^\w\s]', '', regex=True)
df1['Director'] = df1['Director'].str.lower().str.replace('[^\w\s]', '', regex=True)
df1['Genre'] = df1['Genre'].str.lower().str.replace('[^\w\s]', '', regex=True)
df1['Title'] = df1['Title'].str.lower().str.replace('[^\w\s]', '', regex=True)
df1['Year'] = df1['Year'].str.lower().str.replace('[^\w\s]', '', regex=True)
df1['imdbRating'] = df1['imdbRating'].str.lower().str.replace('[^\w\s]', '', regex=True)
df1['Awards'] = df1['Awards'].str.lower().str.replace('[^\w\s]', '', regex=True)


# For each record in the dataframe
for record in df1.to_dict("records"):
    # Update the document with the same 'Title', or insert it if it doesn't exist
    collection.update_one({'Title': record['Title']}, {'$set': record}, upsert=True)