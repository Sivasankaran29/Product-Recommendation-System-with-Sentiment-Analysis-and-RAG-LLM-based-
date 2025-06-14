# 📱 Flipkart Smart Mobile Recommender (Offline RAG + LLM)

An intelligent, feature-aware, review-grounded product recommender system for mobile phones — powered by **over 5,000 real Flipkart reviews**, **local embeddings**, and **offline LLM (Mistral / Gemma)**.

---

## 🔍 Problem Statement

With thousands of product reviews available online, it's hard for buyers to make quick, confident decisions.

This project helps users:
- Filter mobile phones based on **features** like camera, battery, design, etc.
- Apply **sentiment filters** (positive/negative/neutral)
- Ask natural-language questions like _“Is the camera good?”_
- Get answers **based on real user reviews** (via RAG)

---

## 🧠 Tech Stack

| Component | Details |
|----------|---------|
| 🗃️ Data | 5,000+ Flipkart mobile reviews scraped |
| 💬 LLM | [Mistral](https://ollama.com/library/mistral) / Gemma 2B via **Ollama** (Offline) |
| 🔎 RAG | FAISS vector store + HuggingFace embeddings |
| 📊 Sentiment | TextBlob |
| 🧠 Recommendation | Hybrid (SVD + sentiment) |
| 🌐 UI | Streamlit |
| 🧪 Evaluation | RMSE, Precision@K, F1-score, Manual Judgement |

---

## 🚀 Features

- ✅ **Filter by Feature + Sentiment** (e.g., “battery” + “positive”)
- ✅ **Hybrid Recommender**: Ratings + Sentiment
- ✅ **Offline LLM with RAG**: Mistral/Gemma (via Ollama)
- ✅ **Interactive Q&A** based on real reviews
- ✅ **Vectorized Search** using FAISS
- ✅ **Real-Time Recommendations**

---

## 📊 Evaluation Summary

| Metric              | Result |
|---------------------|--------|
| RMSE (Hybrid Rating) | 1.45   |
| Precision@3          | 0.67   |
| Sentiment Accuracy   | 1.00   |
| F1 Score             | 1.00   |
| RAG Answer Quality   | ✔️ Human-evaluated (coherent + grounded) |

---

## 📁 Project Structure
flipkart_recommendation/
├── app.py # Streamlit frontend
├── vector_build.py # Build vector store from reviews
├── evaluation/
│ ├── evaluate.py # RMSE, Precision@K, F1
│ ├── generate_sentiment_test.py
│ └── sentiment_test_labeled.csv
├── data/
│ ├── analyzed_reviews.csv
│ ├── final_processed_reviews.csv
│ └── cleaned_final_project_data.csv
├── models/
│ └── vector_store/
│ ├── index.faiss
│ └── index.pkl
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## ⚙️ How to Run

### ✅ 1. Install Ollama

Download from: https://ollama.com/download  
Then run:

```bash
ollama run gemma:2b
# or
ollama run mistral
✅ 2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
✅ 3. Run the App
bash
Copy
Edit
streamlit run app.py
💡 Sample Queries
“Is the camera good?”

“Why is this product top rated?”

“Is the battery long-lasting?”

“How is the design of this phone?”

📝 Author
Sivasankaran K
