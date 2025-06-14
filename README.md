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

