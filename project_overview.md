# Technical AI Knowledge Assistant

## 📌 Executive Summary
A full-stack, self-hosted AI Knowledge Base and Research Assistant built with **Python, FastAPI, LangGraph, ChromaDB, BM25, SQLite (WAL mode), and Next.js**. 

It enables engineering teams and developers to ingest local documents (PDFs, Markdown, Text) and GitHub repositories, transforming scattered documentation and codebases into an interactive, queryable AI knowledge engine with verifiable source citations.

---

## 🎯 Problem Statement & Core Value Proposition
* **The Problem:** Standard RAG pipelines relying solely on vector search often fail on technical codebases. They struggle with exact keyword lookups (e.g., function names, variable names, error codes) and lack persistent conversation state across multiple user sessions.
* **The Solution:** Combines **Chroma Vector Search** with **BM25 Search** merged via **Reciprocal Rank Fusion (RRF)** to extract Top-K chunks directly into the LLM context.

---

## 🛠️ Retrieval Architecture Flow

```
                      User Query
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    Chroma Vector Search            BM25 Search
    (semantic meaning)          (keyword matching)
             │                           │
             ▼                           ▼
       Ranked Results              Ranked Results
             └─────────────┬─────────────┘
                           │
                           ▼
                          RRF
                           │
                           ▼
                    Combined Ranking
                           │
                           ▼
                      Top-K Chunks
                           │
                           ▼
                          LLM
                           │
                           ▼
                         Answer
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | Next.js (React 19, TypeScript, Tailwind CSS) | Responsive UI with chat threads, citation badges, and ingestion drawer |
| **Backend API** | Python 3.11+, FastAPI, Uvicorn | Async REST endpoints for chat, uploads, repo indexing, and sessions |
| **Orchestration** | LangGraph, LangChain | Stateful execution workflow (Query → Retrieve → Generate → Persist) |
| **Vector DB** | ChromaDB | Dense semantic vector search and embedding storage |
| **Keyword Search** | Rank-BM25 | Sparse lexical search for code symbols and exact matches |
| **Fusion**| Reciprocal Rank Fusion (RRF) | Score unification to select top-k chunks |
| **Database Memory** | SQLite (WAL Mode) | High-concurrency session persistence & message history |

---

## 💼 Resume Bullet Points & Interview Talking Points

### Resume Bullet Points
> **[Project Name] – Self-Hosted AI Research Assistant | FastAPI, Next.js, ChromaDB, LangGraph**
> * Built a full-stack RAG platform with FastAPI and Next.js exposing REST endpoints for document upload, GitHub repo indexing, and chat to query complex technical knowledge bases.
> * Combined ChromaDB vector search with BM25 keyword search and RRF score fusion to retrieve the top-K relevant chunks for every query.
> * Implemented persistent session memory with SQLite (WAL mode) and stateful multi-step graph orchestration using LangGraph.
