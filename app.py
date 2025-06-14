import streamlit as st
import pandas as pd
from textblob import TextBlob
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# ====== FILE PATHS ======
DATA_PATH = "d:/flipkart_recommendation/data/cleaned_final_project_data.csv"
VECTOR_STORE_PATH = "d:/flipkart_recommendation/models/vector_store"

# ====== CONFIGURATION ======
st.set_page_config("📱 Flipkart Smart Recommender", layout="wide")
st.title("📱 Flipkart Recommender (Offline - Mistral via Ollama)")

# ====== LOAD DATA ======
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=["review_text"], inplace=True)
    df["sentiment_type"] = df["review_text"].apply(classify_sentiment)
    return df

def classify_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

df = load_data()

# ====== LOAD VECTOR INDEX ======
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# ====== FILTER REVIEWS BASED ON FEATURE + SENTIMENT ======
def filter_reviews(feature, sentiment, top_n=3):
    df["feature_match"] = df["review_text"].str.lower().str.contains(feature.lower())
    filtered = df[(df["feature_match"]) & (df["sentiment_type"] == sentiment)]
    top = filtered.groupby("product_id").size().reset_index(name="count")
    top = top.sort_values(by="count", ascending=False).head(top_n)
    return top["product_id"].tolist()

# ====== LLM QA FROM REVIEWS (Mistral via Ollama) ======
def ask_local_llm(question, product_ids):
    if not product_ids:
        return "Please filter some products first."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = Ollama(model="mistral", temperature=0.3)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Create a focused question
    prompt = f"""
Based only on customer reviews, answer this question for these products: {', '.join(product_ids)}.

Question: {question}
    """.strip()

    response = rag_chain.invoke({"query": prompt})
    return response["result"]

# ====== SIDEBAR FILTER UI ======
st.sidebar.header("🔍 Filter Products")
feature = st.sidebar.selectbox("Feature", ["camera", "battery", "performance", "display", "design"])
sentiment = st.sidebar.selectbox("Sentiment", ["positive", "neutral", "negative"])
top_n = st.sidebar.slider("Top N Products", 1, 10, 3)

if "filtered_products" not in st.session_state:
    st.session_state.filtered_products = []

if st.sidebar.button("🔎 Recommend"):
    st.session_state.filtered_products = filter_reviews(feature, sentiment, top_n)
    st.success(f"Filtered Products: {', '.join(st.session_state.filtered_products)}")

# ====== QUESTION ANSWERING UI ======
st.markdown("---")
st.header("🧠 Ask a Question Based on Filtered Reviews")

query = st.text_input("Ask something like: Is the camera good? Why is it highly rated?")

if st.button("🤖 Get Answer"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    elif not st.session_state.filtered_products:
        st.warning("Please filter products first.")
    else:
        with st.spinner("Asking Mistral..."):
            answer = ask_local_llm(query, st.session_state.filtered_products)
        st.markdown("### ✅ Answer:")
        st.success(answer)

        st.markdown("---")
        st.subheader("📌 Products Used:")
        for pid in st.session_state.filtered_products:
            st.markdown(f"- `{pid}`")
