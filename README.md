# Mutual Fund Style Classification

**Course:** MF815 — Machine Learning (Final Project)  
**Author:** Ruizhi Xu

## Overview
This project builds a practical pipeline to classify mutual funds into style buckets (e.g., Large Growth, Small Value) using two complementary approaches:
1. Retrieval-Augmented Generation (RAG) — uses chunked fund factsheets, embeddings, and a lightweight prompt to assign styles with sentence-level evidence.
2. Classical ML (Word2Vec → PCA → KNN) — turns each fund’s text into dense vectors and learns decision boundaries with simple, interpretable models (with GridSearchCV tuning).

## Content
### **Data** ./MutualFundSummary.zip/
It contains ~500 mutual fund fact sheets collected from public sources.
The repo expects plain-text summaries under data/summaries/. Use the ingestion step below to convert PDFs → text.

### **Notebook File** ./mf815_final_code.ipynb/
The notebook demonstrates both methods end-to-end (RAG and Word2Vec pipelines), including evaluation.
1. RAG Classification (Evidence-backed)
  * Chunk each fund’s text (CHUNK_SIZE, CHUNK_OVERLAP).
  * Embed chunks and index per fund (FAISS).
  * Retrieve top-k chunks for that fund and prompt a small LLM to select one of the style labels.
  * Aggregate votes (or use a tie-break rule) and keep evidence sentences.

2. Word2Vec → PCA → KNN (Classical ML)
  * Vectorize: train Word2Vec on summaries; average word vectors → document vector per fund.
  * Reduce: apply PCA for stability/compactness.
  * Model: KNN (distance-weighted) with GridSearchCV over k and distance metric.

### **Results** ./mf_815_final_report.pdf/
Confusion matrices, classification reports, and side-by-side comparisons of RAG vs. classical ML are saved under the report.

## Requirments
* Python: 3.9+ recommended
* If you run the RAG pipeline (embeddings + labeling), you’ll need an LLM provider key. Default config uses OpenAI.
  *Set your key as an environment variable before running:
  1. macOS/Linux: export OPENAI_API_KEY="sk-***"
  2. Windows (PowerShell): $Env:OPENAI_API_KEY="sk-***"
  
