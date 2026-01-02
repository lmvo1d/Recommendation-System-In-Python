from data.load_data import load_movielens
from src.als_model import als, predict
from src.visualize import plot_labeled_matrix
from src.recommender import recommend_for_user

def load_movie_names(path="data/ml-100k/u.item"):
    movies = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.split("|")
            movies[int(parts[0]) - 1] = parts[1]
    return movies


print("Loading MovieLens data...")
ratings = load_movielens()

print("Training ALS model...")
U, V = als(ratings, k=20, lambda_=0.1, iterations=15)

predicted = predict(U, V)

plot_labeled_matrix(predicted[:10, :10],
                    "MovieLens Predicted Ratings (Sample)")

user_id = 10
recommended_movies = recommend_for_user(predicted, user_id, ratings)

print("Top recommendations:", recommended_movies)

movies = load_movie_names()

for mid in recommended_movies:
    print(movies[mid])


