"""
RAG Resume Chatbot - Ingestion Script
Parses PDF resume, chunks text, embeds with Sentence Transformers,
stores in ChromaDB, and exports to vectors.json for Vercel deployment.
"""

import os
import json
import re
import sys
import io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


try:
    from PyPDF2 import PdfReader
except ImportError:
    print("Error: PyPDF2 not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("Error: chromadb not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


# ── Configuration ──────────────────────────────────────────────────────────
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between chunks
COLLECTION_NAME = "resume"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "rizal's_resume.pdf"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
OUTPUT_PATH = PROJECT_ROOT / "data" / "vectors.json"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    print(f"[PDF] Reading PDF: {pdf_path.name}")
    reader = PdfReader(str(pdf_path))
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
        print(f"   Page {i + 1}: {len(page_text or '')} chars extracted")
    print(f"   Total: {len(text)} characters\n")
    return text


def clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace and remove artifacts."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common PDF artifacts
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    print(f"[CHUNK] Chunking text (size={chunk_size}, overlap={overlap})")
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end near the chunk boundary
            last_period = text.rfind('.', start + chunk_size // 2, end + 50)
            if last_period > start:
                end = last_period + 1

        chunk_text_content = text[start:end].strip()

        if chunk_text_content:  # Skip empty chunks
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text_content,
                "metadata": {
                    "source": "resume",
                    "chunk_index": chunk_id,
                    "start_char": start,
                    "end_char": end,
                }
            })
            chunk_id += 1

        start = end - overlap

    print(f"   Created {len(chunks)} chunks\n")
    return chunks


def store_in_chromadb(chunks: list[dict]) -> chromadb.Collection:
    """Store chunks in ChromaDB with Sentence Transformer embeddings."""
    print("[EMBED] Initializing ChromaDB with Sentence Transformers...")
    print(f"   Storage: {CHROMA_DIR}")

    # Initialize ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Use Sentence Transformers for embeddings
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete existing collection if it exists, then create fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   Deleted existing '{COLLECTION_NAME}' collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Add chunks to collection
    print(f"   Adding {len(chunks)} chunks to collection...")
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"   [OK] Stored {collection.count()} chunks in ChromaDB\n")
    return collection


def export_vectors(collection: chromadb.Collection, output_path: Path):
    """Export ChromaDB collection to JSON for Vercel deployment."""
    print(f"[EXPORT] Exporting vectors to {output_path}")

    # Get all data from collection including embeddings
    results = collection.get(
        include=["documents", "embeddings", "metadatas"]
    )

    # Build export structure
    embeddings = results["embeddings"]
    has_embeddings = embeddings is not None and len(embeddings) > 0
    export_data = {
        "model": "all-MiniLM-L6-v2",
        "dimension": len(embeddings[0]) if has_embeddings else 0,
        "total_chunks": len(results["ids"]),
        "chunks": []
    }

    for i, chunk_id in enumerate(results["ids"]):
        # Convert numpy array to list for JSON serialization
        emb = embeddings[i]
        emb_list = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
        export_data["chunks"].append({
            "id": chunk_id,
            "text": results["documents"][i],
            "embedding": emb_list,
            "metadata": results["metadatas"][i],
        })

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    file_size = output_path.stat().st_size / 1024
    print(f"   [OK] Exported {export_data['total_chunks']} chunks ({file_size:.1f} KB)")
    print(f"   Embedding dimension: {export_data['dimension']}\n")


def main():
    print("=" * 60)
    print("  RAG Resume Chatbot - Ingestion Pipeline")
    print("=" * 60 + "\n")

    # 1. Check PDF exists
    if not PDF_PATH.exists():
        print(f"[ERROR] PDF not found: {PDF_PATH}")
        print("   Make sure your resume PDF is in the project root.")
        sys.exit(1)

    # 2. Extract text from PDF
    raw_text = extract_text_from_pdf(PDF_PATH)

    if not raw_text.strip():
        print("[ERROR] No text extracted from PDF. The PDF might be image-based.")
        print("   Consider using OCR (e.g., pytesseract) for scanned documents.")
        sys.exit(1)

    # 3. Clean and chunk text
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned)

    # Print sample chunk
    if chunks:
        print("[SAMPLE] Sample chunk (first):")
        print(f"   \"{chunks[0]['text'][:150]}...\"\n")

    # 4. Store in ChromaDB
    collection = store_in_chromadb(chunks)

    # 5. Export to JSON for Vercel
    export_vectors(collection, OUTPUT_PATH)

    print("=" * 60)
    print("  [DONE] Ingestion complete!")
    print(f"  Vectors saved to: {OUTPUT_PATH}")
    print(f"  ChromaDB stored at: {CHROMA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
