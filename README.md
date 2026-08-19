# Resume Chatbot (RAG)

An interactive chatbot that answers questions about my background, skills, and experience directly from my resume PDF using RAG.

Built with a blue neobrutalist UI and deployed on Vercel.

---

## How It Works

1. **Ingestion (Python)**: `scripts/ingest.py` parses `rizal's_resume.pdf`, chunks the text, computes embeddings with ChromaDB + Sentence Transformers (`all-MiniLM-L6-v2`), and exports them into `data/vectors.json`.
2. **Retrieval & Answer (Next.js)**: When a question is asked, the API matches relevant sections from the pre-computed data and streams an answer using Groq (`qwen/qwen3.6-27b`).

---

## Tech Stack

- **Framework**: Next.js 15 (App Router) + React 19
- **LLM Provider**: Groq API
- **Vector DB / Embeddings**: ChromaDB, Sentence Transformers (`all-MiniLM-L6-v2`)
- **Styling**: Vanilla CSS (Blue Neobrutalism)
- **Deployment**: Vercel

---

## Getting Started

### 1. Clone & Install Dependencies

```bash
# Node dependencies
npm install

# Python dependencies (for ingestion)
pip install -r scripts/requirements.txt
```

### 2. Configure Environment

Create a `.env.local` file in the root directory:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free API key at [console.groq.com](https://console.groq.com).

### 3. Build / Update the Resume Index (Optional)

If you update the resume PDF:

```bash
python scripts/ingest.py
```

This will re-chunk and update `data/vectors.json`.

### 4. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---