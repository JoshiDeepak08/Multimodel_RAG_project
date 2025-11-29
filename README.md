# Multimodal RAG — Offline Chatbot & Document RAG Index

A lightweight multimodal Retrieval-Augmented Generation (RAG) backend that ingests documents, images and audio, extracts text (OCR / transcription), stores text chunks in SQLite, builds FAISS embeddings (using `sentence-transformers`), and serves a minimal Flask API for upload, indexing and semantic query.

This project is designed for offline / self-hosted RAG workflows — great for private document collections, internal knowledge bases and prototyping grounded LLM answers.

---

## Live demo

The tool is deployed and live on Hugging Face Spaces:  
**https://huggingface.co/spaces/joshi-deepak08/RAG_based_offline_chatbot**

---

## What it does (high level)

- Accepts uploads (PDF, DOCX, TXT, common image/audio formats).  
- Extracts text:
  - PDFs → text + embedded images (via `PyMuPDF`)  
  - DOCX/DOC/TXT → plain text  
  - Images → OCR (Tesseract)  
  - Audio → offline transcription with `faster-whisper` (if available) or OpenAI Whisper (if `OPENAI_API_KEY` set)  
- Stores metadata & chunks in a local SQLite (`rag_local.db`).  
- Chunks text and builds a FAISS index with `sentence-transformers` embeddings (`intfloat/e5-small-v2` used by default).  
- Provides endpoints to upload files, build the index, list documents and perform semantic queries.  
- Optionally generates grounded summaries via OpenAI chat completion (if `OPENAI_API_KEY` set).

---

## Quick start (local)

> Tested with Python 3.10+. Use a virtual environment.

1. **Clone**

```bash
git clone https://github.com/JoshiDeepak08/<repo-name>.git
cd <repo-name>
