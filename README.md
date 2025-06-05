<p align="center">
    <img src="logo_walkrag.png" alt="walkRAG Logo">
</p>

# WalkRAG

**WalkRAG** is a spatially-enhanced Retrieval-Augmented Generation (RAG) framework for recommending walkable urban itineraries. It integrates Large Language Models (LLMs) with spatial reasoning and geographic data to support exploratory walking and urban discovery through a conversational interface.

This project is associated with the paper:

> **Spatially-Enhanced Retrieval-Augmented Generation for Walkability and Urban Discovery**  
> Maddalena Amendola*, Chiara Pugliese*, Raffaele Perego, Chiara Renso  
> *Both authors contributed equally to this research.  

---

## 🚶 What is WalkRAG?

WalkRAG helps users explore cities by generating personalized walking routes that prioritize:

* **Walkability** (e.g., sidewalks, air quality, greenery, accessibility)
* **User preferences** (e.g., types of POIs or route aesthetics)
* **Contextual information** (e.g., fun facts, history, or details about places encountered along the route)

The system combines three main components:

* **QUAG**: Query understanding and answer generation via an LLM (Llama 3.1 8B)
* **Spatial Component**: Computes walkable routes and scores them based on spatial indicators
* **IR Component**: Retrieves relevant context passages using a dense vector index (FAISS + Snowflake bi-encoder)

---

## 🔍 Key Features

* **Natural language queries** for both routes and urban information
* **Multi-criteria walkability assessment** using OpenStreetMap and environmental data
* **Retrieval-augmented generation** to mitigate hallucinations and improve accuracy
* **Custom dataset** for evaluating spatial and information-seeking queries

---

## 📂 Repository Structure

* `main.py` – sets up and runs the interactive WalkRAG system
* `QUAG.py` – LLM-based query classification and generation
* `src/spatial_component/` – Route generation, walkability scoring, and POI enrichment
* `src/RAG_system/` – Dense passage indexing and neural search
* `dataset/` – WalkRAG evaluation dataset (10 route queries + 30 follow-ups) 
* `output/` – WalkRAG outputs
---

## 📊 Evaluation Summary

| Query Type      | Metric         | WalkRAG | LLM-Only |
| --------------- | -------------- | ------- | -------- |
| Spatial Queries | Fully Correct  | 4 / 10  | 0 / 10   |
|                 | Partially Corr | 6 / 10  | 0 / 10   |
| Info Queries    | Fully Correct  | 20 / 30 | 12 / 30  |
|                 | Partially Corr | 5 / 30  | 11 / 30  |

WalkRAG clearly outperforms closed-book LLMs in both spatial understanding and factual accuracy.

