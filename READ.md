# HR Policy Bot – Agentic AI Capstone Project

## Overview
An intelligent HR policy assistant that answers employee questions using RAG (Retrieval-Augmented Generation), maintains conversation memory, self-evaluates its answers, and never hallucinates. Built with LangGraph, ChromaDB, and GitHub Models.

## Features
- **RAG** – Answers only from 10 HR policy documents
- **Memory** – Remembers user name and conversation context within a session
- **Self‑Evaluation** – Checks faithfulness (0–1) and retries if below 0.7
- **Tool Use** – Provides current date/time
- **Web Interface** – Chat UI built with Streamlit

## Tech Stack
- LangGraph – State machine / agent orchestration
- ChromaDB – Vector store for policy retrieval
- Sentence‑Transformers (all‑MiniLM‑L6‑v2) – Embeddings
- GitHub Models (gpt‑4o‑mini) – LLM via OpenAI‑compatible API
- Streamlit – Frontend chat interface

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/hr-assistant.git
cd hr-assistant