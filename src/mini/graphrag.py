# pip install graphrag  - Requires-Python <3.13,>=3.10  <= 3.12.11
# pip install pypandoc unstructured pymupdf pymilvus
# ollama pull mistral
# ollama pull nomic-embed-text
# PDF/DOCX => Markdown => chunking => embedding (nomic-embed-text) => Milvus => GraphRAG query Ollama (Mistral)
import os
import pypandoc
from graphrag import GraphRAG
from graphrag.chunk import text_to_chunks

# Convert PDF -> Markdown
input_file = "data/data_files/sample.pdf"
md_text = pypandoc.convert_file(input_file, "md")

# Chunking
chunks = text_to_chunks(md_text, chunk_size=500, overlap=50)
documents = [{"id": f"chunk_{i}", "content": chunk} for i, chunk in enumerate(chunks)]

print(f"[INFO] Generated {len(documents)} chunks")
      
# Init GraphRAG with config (Ollama + Milvus)
rag = GraphRAG(config_file="settings.yaml")

# Index documents (store embeddings to Milvus)
rag.index_many(documents)
print("[INFO] Documents indexed into Milvus")

# Ask a question
query = "Summarize the main content of the document?"  # Enhance with using Query Rewriter

response = rag.query(query)

print("Q:", query)
print("A:", response)
