# Multi-Format RAG AI System

Hey there! 👋 This is a **RAG (Retrieval-Augmented Generation)** system that lets you chat with your documents. Upload PDFs, Excel files, Word docs, or CSVs, and ask questions in plain English. The AI reads your files, understands them, and answers your questions with actual citations!

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![React](https://img.shields.io/badge/React-19-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

## 🎯 What Does This Do?

Imagine you have a bunch of Excel sheets with financial data, or PDFs with research papers. Instead of manually searching through them, you just ask: **"What was the total revenue in Q1?"** or **"Summarize the methodology section"** — and boom, the AI finds the info and tells you, with sources!

## 🤔 Why Did We Build It This Way?

### **The Problem We Solved**

Regular chatbots like ChatGPT can't see your private files. And even if you copy-paste data, they sometimes make up answers (we call this "hallucination"). So we built a system that:
1. **Actually reads your files** (not just guesses)
2. **Cites sources** (like "this answer comes from row 15 of sales.xlsx")
3. **Handles different file types** (PDF text vs Excel numbers need different treatment)
4. **Runs calculations** (for Excel, it writes Python code on-the-fly!)

### **Why RAG (Retrieval-Augmented Generation)?**

Think of RAG like an open-book exam. Instead of making the AI memorize everything (which doesn't work for your private files), we:
1. **Store** your documents in a searchable database
2. **Retrieve** relevant parts when you ask a question
3. **Generate** an answer based on what we found

This way, answers are always grounded in your actual data, not made up!

## 🏗️ How It Actually Works (Simple Version)

```
You upload "sales.xlsx"
    ↓
We break it into chunks (rows 1-50, 51-100, etc.)
    ↓
Convert chunks to numbers (vectors) that AI understands
    ↓
Store in a vector database (ChromaDB)
    ↓
You ask: "Who spent the most?"
    ↓
We convert your question to a vector
    ↓
Find the most similar chunks (semantic search)
    ↓
AI writes Python code: df.groupby('name')['amount'].sum().max()
    ↓
Execute the code on your real Excel file
    ↓
Format the answer: "John Doe spent $5,000 [1]"
```

## 💡 What Makes This Special?

### **1. Hybrid Approach: RAG + Code Execution**

Most RAG systems only do text search. We do **both**:
- **For PDFs/Word docs**: Search for relevant paragraphs
- **For Excel/CSV**: Generate and run Python code to calculate answers

**Why?** Because "What's the total?" needs a calculation, not just finding text. We realized that Excel queries are fundamentally different from document queries.

### **2. Smart Query Routing**

We use AI to classify your question first:
- "Summarize this doc" → Text retrieval strategy
- "What's the average?" → Code execution strategy
- "Show all transactions" → Data filtering strategy

**Why?** Not all questions need the same solution. This makes responses faster and more accurate.

### **3. Chunking Without Breaking Context**

When we split documents:
- **PDFs**: 500 words per chunk with 100-word overlap (so we don't cut sentences in half)
- **Excel**: 50 rows per chunk + column headers (so AI knows what each column means)

**Why?** If we chunk too small, we lose context. Too large, and searching becomes slow. We tested different sizes and this worked best.

### **4. Real Citations (Not Fake Ones)**

Every answer has numbered sources like [1], [2]. Click them to see which file it came from.

**Why?** Trust. You can verify if the AI is correct by checking the source.

### **5. Safe Code Execution**

When the AI generates Python code, we:
- Whitelist only safe libraries (pandas, numpy)
- Block dangerous stuff (file deletion, network access)
- Timeout after 30 seconds (no infinite loops)
- Run in an isolated environment

**Why?** Security! We don't want auto-generated code to harm your system.

## 🛠️ Tech Stack & Why We Chose Each

### **Frontend: Next.js 16 + TypeScript**
**Why Next.js?** Server-side rendering means fast page loads. Built-in optimizations. React 19 gives us better performance.

**Why TypeScript?** Catches bugs before runtime. Makes the code easier to understand when you come back to it later.

**Why Tailwind CSS?** You can style things super fast without writing custom CSS files. Plus it's responsive by default.

### **Backend: FastAPI (Python)**
**Why FastAPI?** It's async (handles multiple users at once), auto-generates API docs, and works great with Python ML libraries.

**Why Python?** All the AI/ML tools are in Python. Plus pandas for Excel manipulation is unbeatable.

### **AI: Google Gemini (gemini-2.0-flash-exp)**
**Why Gemini over OpenAI?**
- **Cost**: Cheaper than GPT-4
- **Speed**: "Flash" version is optimized for low latency
- **Free tier**: Generous limits for development
- **Context window**: Can handle long documents

### **Embeddings: text-embedding-004 (Google)**
**Why not use Gemini's embeddings?** Actually, we DO! text-embedding-004 is Google's embedding model. 768 dimensions gives us great accuracy without being too slow.

### **Vector Database: ChromaDB**
**Why ChromaDB?**
- **Easy**: No server setup, just works
- **Fast**: Built for embeddings (finds similar chunks in milliseconds)
- **Persistent**: Saves to disk, survives restarts
- **Free**: Open source

We tried FAISS (too low-level) and Pinecone (requires cloud), but ChromaDB was the sweet spot.

### **Document Processing: PyPDF2, python-docx, openpyxl, pandas**
These are battle-tested libraries. pandas especially is the standard for Excel in Python — everyone uses it, lots of Stack Overflow answers when you're stuck!

## 🔄 The Workflow (Technical)

### **Upload Flow**
```
1. User drops file → Frontend validates (size, type)
2. POST to /api/upload → Backend receives file
3. Document Processor reads file (PDF/Excel/Word/CSV)
4. Extract metadata (for Excel: columns, data types, sample rows)
5. Chunk intelligently (50 rows for Excel, 500 words for text)
6. Generate embeddings (convert text to 768-d vectors)
7. Store in ChromaDB with metadata tags
8. Save original file to disk (so we can run code on it later)
9. Return success + chunk count
```

### **Query Flow**
```
1. User asks: "What's the total revenue?"
2. Query Router (AI) classifies: "COMPUTATIONAL"
3. Embed the query (convert to vector)
4. ChromaDB searches for similar chunks (cosine similarity)
5. Retrieve top 5 most relevant chunks
6. Since it's computational:
   → Excel Code Executor generates Python code
   → Validates code (no dangerous operations)
   → Executes on actual Excel file
   → Gets result: "$1,234,567"
7. AI combines: context + code result
8. Generate natural language answer: "Total revenue was $1,234,567 [1]"
9. Return answer + sources + structured data
```

## 🎨 Design Decisions (The Interesting Stuff)

### **Why Not Just Stuff Everything Into One Prompt?**
We tried that! Problem: AI models have token limits (context windows). A 10,000-row Excel file won't fit. Plus, it's slow and expensive.

**Our solution:** Store everything in a database, only retrieve what's needed for each question.

### **Why Two Different Strategies (Text Search vs Code Generation)?**
We noticed:
- "Summarize this document" → Just need to read text
- "What's the sum of column B?" → Need to calculate

Trying to make one strategy do both made results worse. Specialized strategies work better.

### **Why 500 Words per Chunk?**
We tested 100, 250, 500, 1000 words.
- Too small (100): Loses context, AI can't understand
- Too large (1000): Search becomes imprecise
- 500 words: Goldilocks zone! Enough context, precise search.

### **Why Overlap Chunks?**
Imagine a sentence gets cut in half between chunks. The AI won't understand either half. By overlapping 100 words, important info appears in multiple chunks.

### **Why Not Fine-Tune Our Own Model?**
Fine-tuning requires:
- Thousands of training examples
- Expensive GPU time
- Ongoing maintenance

Using Gemini's pre-trained model:
- Works immediately
- Gets better when Google updates it
- Costs $0 for development

If we had specific domain needs (medical, legal), fine-tuning would make sense. For general documents, pre-trained is smarter.

## 🚀 Quick Start

### **Prerequisites**
- Node.js 18+
- Python 3.10+
- Google Gemini API key ([Get free key](https://makersuite.google.com/app/apikey))

### **Setup (5 minutes)**

1. **Clone**
```bash
git clone https://github.com/yourusername/rag-ai.git
cd rag-ai
```

2. **Backend**
```bash
cd backend
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_actual_key_here" > .env
python app.py
```

3. **Frontend (new terminal)**
```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

4. **Open** http://localhost:3000

## 📦 Deployment

### **Vercel (Frontend)**
1. Push to GitHub
2. Import repo on vercel.com
3. Set `NEXT_PUBLIC_API_URL` to your backend URL
4. Deploy!

### **Railway (Backend)**
1. New project on railway.app
2. Connect GitHub repo
3. Root directory: `backend`
4. Add `GOOGLE_API_KEY` env variable
5. Deploy!

## 🎯 What Can You Ask?

### **For Excel/CSV:**
- "What's the total revenue?"
- "Show all transactions over $1000"
- "Who spent the most money?"
- "What's the average purchase amount?"
- "Filter by date range"

### **For PDFs/Documents:**
- "Summarize this document"
- "What does section 3 say?"
- "Find mentions of 'machine learning'"
- "Compare the two methodologies"

### **Cross-File:**
- "Which file has the highest revenue?"
- "Summarize all uploaded documents"

## 🔧 How to Customize

### **Change Chunk Size**
Edit `backend/config.py`:
```python
CHUNK_SIZE = 500  # words for text
CHUNK_OVERLAP = 100  # word overlap
```

### **Change Embedding Model**
Edit `backend/src/core/rag_engine_v2.py`:
```python
# Use different model
EMBEDDING_MODEL = 'text-embedding-004'  # or 'all-MiniLM-L6-v2'
```

### **Add New File Types**
Add processor in `backend/src/processors/document_processor.py`

## 🐛 Common Issues

**"API key not found"**
→ Did you create `.env` file in backend folder?

**"Failed to fetch documents"**
→ Is backend running on port 8000? Check `NEXT_PUBLIC_API_URL` in frontend.

**"Upload fails for large files"**
→ Default limit is 10MB. Change in `backend/config.py`

**"Slow responses"**
→ First query is always slower (loading models). Subsequent queries are fast.

## 📊 Performance

- **Upload**: ~2-5 seconds for 1000-row Excel
- **Query**: ~2-3 seconds average
- **Embeddings**: 768 dimensions
- **Storage**: ~100KB per document in ChromaDB
- **Max file size**: 10MB (configurable)

## 🎓 Learning Resources

If you want to understand RAG better:
- [RAG Explained (Anthropic)](https://www.anthropic.com/index/retrieval-augmented-generation)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Next.js Docs](https://nextjs.org/docs)

## 🤝 Contributing

Found a bug? Want to add features? PRs welcome!

1. Fork the repo
2. Create a branch (`git checkout -b feature/cool-feature`)
3. Commit (`git commit -m 'Add cool feature'`)
4. Push (`git push origin feature/cool-feature`)
5. Open a Pull Request

## 📄 License

MIT License - use it however you want!

## 🙏 Built With

- **Google Gemini** for AI magic
- **ChromaDB** for vector search
- **FastAPI** for the backend
- **Next.js** for the frontend
- **All the amazing open-source libraries**

---

**Made with ❤️ for people who are tired of manually searching through documents**

Got questions? Open an issue on GitHub!
