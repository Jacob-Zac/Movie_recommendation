from transformers import DistilBertTokenizer, DistilBertModel
import torch
from flask import Flask
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize
from pymongo import MongoClient
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask_caching import Cache
from flask import Flask, request, render_template
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import jsonify

app = Flask(__name__)
@app.route('/search_movie', methods=['GET'])
def search_movie():
    query = request.args.get('term', '').lower()
    df = fetch_movies_data()  # Using the function you have previously defined
    matching_movies = df[df['Title'].str.lower().str.contains(query)]['Title'].tolist()
    return jsonify(matching_movies)

# Use DistilBERT for speed
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

cache_config = {
    "DEBUG": True,
    "CACHE_TYPE": "simple",
}

app.config.from_mapping(cache_config)
cache = Cache(app)

def vectorize_plot(plot):
    """Generate embedding for a plot using DistilBERT."""
    input_ids = tokenizer.encode(plot, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        output = model(input_ids)
    plot_embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return plot_embedding

def preprocess_text(text):
    """Preprocess a text."""
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english') and word not in string.punctuation]
    return ' '.join(words)

@cache.memoize(timeout=3600)
def fetch_movies_data():
    client = MongoClient('mongodb+srv://jacobrz030:oOnUj5jifqgokKZI@movies.2qhakzv.mongodb.net/?retryWrites=true&w=majority')
    db = client['IMDB']
    collection = db['Movies']
    cursor = collection.find()
    df = pd.DataFrame(list(cursor))
    
    # Preprocess and embed once
    df['Processed_Plot'] = df['Plot'].apply(preprocess_text)
    df['plot_vector'] = df['Processed_Plot'].apply(vectorize_plot)
    
    return df

@cache.memoize(timeout=3600)
def recommend_movies(movie_title):
    df = fetch_movies_data()

    # Create TF-IDF features
    combined_fields = df['Title'].astype(str) + ' ' + df['Genre'].astype(str) + ' ' + df['Country'].astype(str) + ' ' + df['Director'].astype(str) + ' ' + df['Actors'].astype(str) + ' ' + df['imdbRating'].astype(str) + ' ' + df['Awards'].astype(str)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(combined_fields)
    tfidf_features = tfidf_matrix.toarray()

    # Combine TF-IDF features with DistilBERT embeddings
    combined_features = np.hstack([tfidf_features, np.array(df['plot_vector'].tolist())])

    # Normalize the combined features
    normalized_features = normalize(combined_features)

    # Compute similarity matrix
    cosine_sim = linear_kernel(normalized_features, normalized_features)

    # Resetting index and preparing for recommendation
    df = df.reset_index(drop=True)
    indices = pd.Series(df.index, index=df['Title'])

    # Check if movie exists
    if movie_title not in df['Title'].values:
        return "The movie title does not exist in the database."

    idx = indices[movie_title]
    cosine_sim_scores = list(enumerate(cosine_sim[idx]))
    cosine_sim_scores = sorted(cosine_sim_scores, key=lambda x: x[1], reverse=True)[1:5]  # Exclude the movie itself

    # Extract recommended movie info
    movie_info = [(df['Title'].iloc[i[0]], i[1], df['Poster'].iloc[i[0]]) for i in cosine_sim_scores]

    return movie_info

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    recommendations = []
    if request.method == 'POST':
        movie_title = request.form['movie_title'].lower() 
        recommendations = recommend_movies(movie_title)
    return render_template('home.html', recommendations=recommendations)  # Assuming the provided HTML is named movie_recommendation.html

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5500)