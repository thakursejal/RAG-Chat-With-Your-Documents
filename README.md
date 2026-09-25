# 📚 Chat With Your Documents — RAG AI Assistant

An AI-powered document question-answering application built using Retrieval-Augmented Generation (RAG), embeddings, ChromaDB and Hugging Face LLMs.

## 🎯 Project Overview

Chat With Your Documents allows users to upload a PDF and ask questions about its content.

The application retrieves relevant information from the uploaded document and provides it to an LLM as context before generating the final answer.

## 🔄 RAG Workflow

Document → Text Extraction → Text Chunking → Embeddings → ChromaDB → Similarity Search → Relevant Context → LLM → Final Answer

## ✨ Features

- Upload PDF documents
- Automatic text extraction
- Document chunking
- Sentence Transformer embeddings
- ChromaDB vector database
- Semantic similarity search
- Hugging Face LLM integration
- Document-grounded answers
- View retrieved context
- Simple Streamlit interface

## 🛠️ Technologies Used

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- ChromaDB
- Hugging Face
- GitHub

## 🧠 Embedding Model

all-MiniLM-L6-v2

The model converts text into numerical vectors that can be compared using semantic similarity.

## 🤖 Language Model

openai/gpt-oss-120b:fastest

The model receives relevant document context and generates an answer based on that context.

## 🚀 How It Works

1. User uploads a PDF.
2. Text is extracted from the document.
3. The text is divided into smaller chunks.
4. Chunks are converted into embeddings.
5. Embeddings are stored in ChromaDB.
6. The user's question is converted into an embedding.
7. Similar document chunks are retrieved.
8. Retrieved context is sent to the LLM.
9. The LLM generates the final answer.

## 🔐 Security

The Hugging Face API token is stored securely as a Streamlit secret.

The token is not included in the GitHub repository.

## 📚 Learning Outcomes

This project demonstrates:

- LLM hallucinations
- Retrieval-Augmented Generation
- Text embeddings
- Semantic similarity
- Document processing
- Text chunking
- Vector databases
- Similarity search
- Context retrieval
- LLM integration
- Document-based question answering

## 👩‍💻 Author

Thakur Sejal

B.Tech — Artificial Intelligence & Machine Learning

Codomax Digital Solutions Internship

Module 4 — RAG, Embeddings & Vector Databases
