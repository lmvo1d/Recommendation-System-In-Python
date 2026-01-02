def recommend_for_user(predicted, user_id, known_ratings, top_n=5):
    scores = predicted[user_id]
    scores[known_ratings[user_id] > 0] = -1

    return scores.argsort()[::-1][:top_n]
