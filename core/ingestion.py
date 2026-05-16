import os
import fitz
from typing import List, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    pages = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({"page_number": page_num + 1, "text": text.strip()})
    doc.close()
    print(f"[Ingestion] Loaded {len(pages)} pages from: {os.path.basename(file_path)}")
    return pages

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

def ingest_pdf(file_path: str) -> List[DocumentChunk]:
    filename = os.path.basename(file_path)
    pages = load_pdf(file_path)
    all_chunks: List[DocumentChunk] = []
    chunk_index = 0
    for page_data in pages:
        page_chunks = chunk_text(page_data["text"])
        for chunk_text_content in page_chunks:
            all_chunks.append(DocumentChunk(
                text=chunk_text_content,
                source=filename,
                page=page_data["page_number"],
                chunk_index=chunk_index,
                metadata={"source": filename, "page": page_data["page_number"], "chunk_index": chunk_index}
            ))
            chunk_index += 1
    print(f"[Ingestion] Created {len(all_chunks)} chunks from {filename}")
    return all_chunks

def ingest_directory(dir_path: str) -> List[DocumentChunk]:
    if not os.path.exists(dir_path):
        raise NotADirectoryError(f"Directory not found: {dir_path}")
    all_chunks = []
    pdf_files = [f for f in os.listdir(dir_path) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"[Ingestion] No PDF files found in {dir_path}")
        return []
    for pdf_file in pdf_files:
        full_path = os.path.join(dir_path, pdf_file)
        chunks = ingest_pdf(full_path)
        all_chunks.extend(chunks)
    print(f"[Ingestion] Total chunks from directory: {len(all_chunks)}")
    return all_chunks
