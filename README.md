# Multi-Format RAG AI System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![React](https://img.shields.io/badge/React-19-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

I built this system to solve a simple pain point: **I wanted to query my own documents the same way I query an AI model**, but with answers grounded in actual data — not guesses. This project lets me upload PDFs, Word docs, Excel sheets, or CSVs and ask questions that require searching, summarizing, or even calculations.

## What the system does

Once I upload a file, I can ask things like:

* “Summarize section 3 of this PDF.”
* “Calculate the total revenue in this Excel file.”
* “Filter all rows where amount > 1000.”
* “Find references to machine learning across all documents.”

The system retrieves only the relevant chunks, performs text or numeric operations when needed, and returns an answer with proper citations.

## How I designed it

I approached the design with two goals: **accuracy** and **speed**.

* **RAG for text** → semantic search over PDFs, DOCX, etc.
* **Python code execution for spreadsheets** → real calculations, not approximations.
* **Query router** → classifies each question (text retrieval, computation, filtering, etc.).
* **Chunking tuned for context** → text split into ~500-word windows, Excel split by rows with preserved headers.
* **ChromaDB** → fast, lightweight vector store that works well for local and cloud environments.

This setup avoids hallucination because every answer comes from a retrieved chunk or an executed computation.

## Internal workflow

**File upload → Processing → Embeddings → Semantic search → (Optional) Code execution → Final answer**

Under the hood:

1. Extract text/rows using PyPDF2, python-docx, openpyxl, pandas
2. Chunk with overlap (text) or row windows (spreadsheets)
3. Generate 768-dim embeddings using Google’s `text-embedding-004`
4. Store chunks in ChromaDB with metadata pointing to file + chunk range
5. When I ask something:

   * The router classifies the intent
   * Relevant chunks are retrieved
   * If needed, safe Python code is generated + executed
   * The final response is composed with citations

The system never executes unsafe code — only whitelisted imports, restricted namespace, and isolated runtime.

## Tech stack

* **Frontend:** Next.js 16, React 19, Tailwind
* **Backend:** FastAPI (async), Python 3.10+
* **AI:** Google Gemini (Flash models)
* **Embeddings:** text-embedding-004
* **Vector DB:** ChromaDB
* **Document processing:** pandas, PyPDF2, python-docx, openpyxl

This combination gave me the best balance of speed, cost-efficiency, and flexibility.

## Running locally

```bash
# Backend
cd backend
pip install -r requirements.txt
python app.py
```

```bash
# Frontend
cd frontend
npm install
npm run dev
```

Open: **[http://localhost:3000](http://localhost:3000)**

## Example queries I use daily

* “Top 10 highest transactions in this file.”
* “Give me a quick summary of all documents.”
* “Compare section 1 of file A with section 2 of file B.”
* “Which customer has the highest spend?”
* “Extract all dates and sort them.”

## License

MIT — feel free to adapt or extend it for your own workflows.