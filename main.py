import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample movie data (movie_id, title, genres)
movies_data = [
    (1, "Movie A", "Action|Adventure|Sci-Fi"),
    (2, "Movie B", "Action|Drama|Thriller"),
    (3, "Movie C", "Comedy|Romance"),
    (4, "Movie D", "Action|Adventure"),
    (5, "Movie E", "Comedy|Family"),
]

# Sample user preferences (user_id, movie_id, rating)
user_preferences = [
    (1, 1, 5),
    (1, 2, 4),
    (2, 1, 3),
    (2, 3, 5),
]

# Convert data to dataframes
movies_df = pd.DataFrame(movies_data, columns=["movie_id", "title", "genres"])
ratings_df = pd.DataFrame(user_preferences, columns=["user_id", "movie_id", "rating"])

# Merge dataframes to include genres in ratings
movie_ratings_df = pd.merge(ratings_df, movies_df, on="movie_id")
# Create a TF-IDF Vectorizer to convert genres into numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df["genres"])

# Calculate the cosine similarity between movies based on genres
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on user preferences
def get_movie_recommendations(user_id):
    user_movies = movie_ratings_df[movie_ratings_df["user_id"] == user_id]
    user_genres = " ".join(user_movies["genres"].values)

    # Transform user's genres into TF-IDF representation
    user_tfidf = tfidf_vectorizer.transform([user_genres])

    # Calculate cosine similarity between user's preferences and all movies
    similarity_scores = linear_kernel(user_tfidf, tfidf_matrix).flatten()

    # Get movie indices sorted by similarity score
    movie_indices = similarity_scores.argsort()[::-1]

    # Exclude movies the user has already rated
    recommended_movies = [movie for movie in movie_indices if movie not in user_movies["movie_id"].values]

    return movies_df.iloc[recommended_movies]["title"]

# Get user input for user ID
user_id_to_recommend = int(input("Enter your user ID: "))
recommendations = get_movie_recommendations(user_id_to_recommend)

print(f"Movie recommendations for User {user_id_to_recommend}:\n{recommendations}")
