# Movie Recommendation System using SVD

This project demonstrates how **linear algebra** powers recommendation systems through **Singular Value Decomposition (SVD)**.

## 🔹 Features
- Collaborative filtering
- Dimensionality reduction
- Rating prediction
- Top-N recommendations
- Matrix visualization

## 🔹 How It Works
1. Build a user–item matrix
2. Apply SVD
3. Keep top latent factors
4. Reconstruct matrix
5. Recommend highest-rated items

## 🔹 Run the Project
```bash
pip install -r requirements.txt
python main.py

## Dataset
This project uses the MovieLens 100K dataset by GroupLens.

## Algorithm
- Alternating Least Squares (ALS)
- Regularized matrix factorization
- Sparse matrix handling

## Results
- Predicts unseen movie ratings
- Generates personalized recommendations
- Scales to real-world datasets
