import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Movie dataset
movies = [
"Batman fights crime in Gotham",
"Space explorers travel through a wormhole",
"A hacker discovers reality is a simulation",
"Dream invasion inside the human mind",
"A superhero protects the city"
]

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(movies)

# Calculate similarity matrix
similarity = cosine_similarity(X)

# Function to recommend movies
def recommend(movie_index):
    scores = list(enumerate(similarity[movie_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    for i in scores[1:3]:
        print(movies[i[0]])
        
# Example: recommend movies similar to movie 0
recommend(0)