import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv(dotenv_path="D:/flipkart_recommendation/.env")

# === PATHS ===
DATA_PATH = "D:/flipkart_recommendation/data/analyzed_reviews.csv"
VECTOR_STORE_PATH = "D:/flipkart_recommendation/models/vector_store"

# === STEP 1: Load & Clean Data ===
df = pd.read_csv(DATA_PATH)
df.drop_duplicates(inplace=True)
df.dropna(subset=["product_id", "user_id", "review_text", "rating"], inplace=True)

# === STEP 2: Normalize IDs ===
df["product_id"] = df["product_id"].astype(str)
df["user_id"] = df["user_id"].astype(str)

# === STEP 3: Sentiment Analysis ===
def get_sentiment_polarity(text):
    return TextBlob(text).sentiment.polarity

df["sentiment_score"] = df["review_text"].apply(get_sentiment_polarity)
df["sentiment_type"] = df["sentiment_score"].apply(
    lambda x: "positive" if x > 0.1 else "negative" if x < -0.1 else "neutral")

# === STEP 4: Collaborative Filtering with SVD ===
ratings_matrix = df.pivot_table(index="user_id", columns="product_id", values="rating", fill_value=0)

n_components = min(20, min(ratings_matrix.shape) - 1)  # dynamically adjust based on matrix size
svd = TruncatedSVD(n_components=n_components)
latent_matrix = svd.fit_transform(ratings_matrix)
similarity_matrix = cosine_similarity(latent_matrix)

user_similarity_df = pd.DataFrame(similarity_matrix, index=ratings_matrix.index, columns=ratings_matrix.index)

def recommend_products(user_id, top_n=5):
    if user_id not in user_similarity_df.index:
        return []
    sim_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:top_n+1].index
    sim_ratings = ratings_matrix.loc[sim_users].mean().sort_values(ascending=False)
    already_rated = ratings_matrix.loc[user_id][ratings_matrix.loc[user_id] > 0].index
    return sim_ratings.drop(already_rated, errors='ignore').head(top_n).index.tolist()

# === STEP 5: Hybrid Recommendation (Ratings + Sentiment) ===
def recommend_hybrid(user_id, top_n=5):
    if user_id not in user_similarity_df.index:
        return []
    sim_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:top_n+1].index
    sim_ratings = ratings_matrix.loc[sim_users].mean().reset_index()
    sim_ratings.columns = ["product_id", "avg_rating"]

    avg_sentiment = df.groupby("product_id")["sentiment_score"].mean().reset_index()
    avg_sentiment.columns = ["product_id", "avg_sentiment"]

    hybrid_df = pd.merge(sim_ratings, avg_sentiment, on="product_id")
    hybrid_df["hybrid_score"] = 0.7 * hybrid_df["avg_rating"] + 0.3 * hybrid_df["avg_sentiment"]

    already_rated = ratings_matrix.loc[user_id][ratings_matrix.loc[user_id] > 0].index
    hybrid_df = hybrid_df[~hybrid_df["product_id"].isin(already_rated)]

    return hybrid_df.sort_values(by="hybrid_score", ascending=False).head(top_n)["product_id"].tolist()

# === STEP 6: RAG Vector Store Embedding ===
grouped = df.groupby("product_id")["review_text"].apply(lambda x: " ".join(x)).reset_index()
docs = [Document(page_content=row["review_text"], metadata={"product_id": row["product_id"]}) for _, row in grouped.iterrows()]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embeddings)
vectorstore.save_local(VECTOR_STORE_PATH)
print("✅ Vector store saved successfully at:", VECTOR_STORE_PATH)

# === Optional: Save intermediate data ===
df.to_csv("D:/flipkart_recommendation/data/final_processed_reviews.csv", index=False)
print("✅ Preprocessed review data saved.")
