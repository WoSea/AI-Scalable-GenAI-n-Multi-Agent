# pip install weaviate-client rdflib datasets tqdm python-dotenv requests sentence-transformers torch
# Install Weaviate local
# Pull LLM & Embeddings local from Ollama 
"""
  Load documents (datasets wiki_snippets or fallback)
  Chunk documents (simple sliding window by words + overlap)
  Build small RDF knowledge graph (rdflib) and run a SPARQL filter
  Create Weaviate collection (optionally configure Ollama vectorizer & generative integration)
  Upsert chunked documents into Weaviate (either letting Weaviate vectorize or supply vectors)
  Hybrid retrieval (weaviate nearText + vector or generative RAG using weaviate.generate)
  Compose final prompt and optionally call Ollama or Mistral directly
"""

import os
import json
import time
import requests
from typing import List, Tuple, Dict, Any

from datasets import load_dataset
from rdflib import Graph, Literal, Namespace, RDF
from tqdm import tqdm

import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.config import Property, DataType, Configure as Conf

from sentence_transformers import SentenceTransformer

# CONFIG
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_CLASS = "DocChunk"
WEAVIATE_VECTORIZE_WITH_OLLAMA = True   # If True -> create collection with text2vec_ollama and don't provide vectors
WEAVIATE_OLLAMA_ENDPOINT = os.getenv("OLLAMA_API", "http://host.docker.internal:11434")
WEAVIATE_OLLAMA_EMBED_MODEL = "nomic-embed-text"   # used if WEAVIATE_VECTORIZE_WITH_OLLAMA == True

USE_OLLAMA_LOCAL_FOR_LLM = True   # If True use Ollama local HTTP API for generation
OLLAMA_LLM_MODEL = "mistral"      # model name in Ollama ('mistral' or 'llama3.2')
USE_MISTRAL_API = False           # If True (and USE_OLLAMA_LOCAL_FOR_LLM False) call remote Mistral API (set key below)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

EMBED_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local embedding fallback
CHUNK_MAX_WORDS = 120
CHUNK_OVERLAP_WORDS = 20

TOP_K = 3
HYBRID_ALPHA = 0.5  # when using weaviate hybrid: 0 => BM25 only, 1 => vector only

# Utilities: chunking
def chunk_text_by_words(text: str, max_words: int = CHUNK_MAX_WORDS, overlap: int = CHUNK_OVERLAP_WORDS
                       ) -> List[Tuple[str, int, int]]:
    """
    Split text into chunks by words with a sliding window and overlap.
    Returns a list of tuples: (chunk_text, start_word_idx, end_word_idx)

    Every original doc is split into smaller chunks and each chunk is inserted as a separate object 
    (with metadata source_doc, chunk_start, chunk_end)
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        start = i
        end = min(i + max_words, n)
        chunk = " ".join(words[start:end])
        chunks.append((chunk, start, end))
        if end == n:
            break
        i = end - overlap  # slide with overlap
    return chunks

# Utilities: embeddings
_local_embedder = None

def init_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        _local_embedder = SentenceTransformer(EMBED_LOCAL_MODEL)
    return _local_embedder

def embed_texts_local(texts: List[str]) -> List[List[float]]:
    """
    Use sentence-transformers to produce embeddings locally.
    """
    model = init_local_embedder()
    embs = model.encode(texts, normalize_embeddings=True)
    return embs.tolist()

def embed_with_ollama_api(texts: List[str]) -> List[List[float]]:
    """
    Call Ollama local embeddings endpoint directly.
    Expects Ollama running locally at WEAVIATE_OLLAMA_ENDPOINT or OLLAMA_BASE.
    """
    out = []
    url = f"{WEAVIATE_OLLAMA_ENDPOINT.rstrip('/')}/api/embeddings"
    for t in texts:
        payload = {"model": WEAVIATE_OLLAMA_EMBED_MODEL, "prompt": t}
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        j = r.json()
        vec = j.get("embedding") or j.get("data", [{}])[0].get("embedding")  # support variations
        out.append(vec)
    return out

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Main embedding entrypoint. If WEAVIATE_VECTORIZE_WITH_OLLAMA is True, often rely on Weaviate
    to vectorize server-side. This function is used when needing local vectors (WEAVIATE_VECTORIZE_WITH_OLLAMA == False)
    or for the query vector when using client-side LLM.
    """
    # prefer Ollama local embeddings if available
    try:
        return embed_with_ollama_api(texts)
    except Exception:
        # fallback to local sentence-transformers
        return embed_texts_local(texts)

# Build RDF graph utilities
def build_rdf_graph_for_docs(first_doc_text: str):
    """
    Create a tiny RDF graph for demonstration and write TTL to disk.
    """
    g = Graph()
    EX = Namespace("http://example.org/")
    g.bind("ex", EX)

    g.add((EX["Doc1"], RDF.type, EX["Policy"]))
    g.add((EX["Doc1"], EX["category"], Literal("Healthcare")))
    g.add((EX["Doc1"], EX["region"], Literal("US")))
    g.add((EX["Doc1"], EX["text"], Literal(first_doc_text)))

    os.makedirs("src/data", exist_ok=True)
    g.serialize(destination="src/data/knowledge.ttl", format="turtle")
    return g, EX

def query_rdf_for_policy_texts(g: Graph, EX: Namespace) -> List[str]:
    """
    Query RDF graph for Policy texts with category=Healthcare and region=US
    """
    sparql = """
    PREFIX ex: <http://example.org/>
    SELECT ?text WHERE {
      ?doc a ex:Policy ;
           ex:category "Healthcare" ;
           ex:region "US" ;
           ex:text ?text .
    }
    """
    results = g.query(sparql)
    return [str(row.text) for row in results]

# Weaviate helpers (new API)
def connect_weaviate_client() -> weaviate.WeaviateClient:
    """
    Connect to a local Weaviate instance with helper connect_to_local per quickstart.
    """
    client = weaviate.connect_to_local(url=WEAVIATE_URL)
    return client

def create_weaviate_collection(client: weaviate.WeaviateClient, class_name: str):
    """
    Create a collection (class) in Weaviate. If WEAVIATE_VECTORIZE_WITH_OLLAMA is True,
    configure the Ollama embedding integration server-side (Weaviate will vectorize automatically).
    Otherwise create a generic class with vectorizer disabled (will pass vector manually).
    """
    # Delete existing class if present
    try:
        if client.collections.exists(class_name):
            client.collections.delete(class_name)
            time.sleep(0.5)
    except Exception:
        pass

    if WEAVIATE_VECTORIZE_WITH_OLLAMA:
        vec_conf = Configure.Vectors.text2vec_ollama(
            api_endpoint=WEAVIATE_OLLAMA_ENDPOINT,
            model=WEAVIATE_OLLAMA_EMBED_MODEL
        )
        gen_conf = Configure.Generative.ollama(
            api_endpoint=WEAVIATE_OLLAMA_ENDPOINT,
            model=OLLAMA_LLM_MODEL
        )
        client.collections.create(
            name=class_name,
            vector_config=vec_conf,
            generative_config=gen_conf,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source_doc", data_type=DataType.TEXT),
                Property(name="chunk_start", data_type=DataType.INT),
                Property(name="chunk_end", data_type=DataType.INT),
                Property(name="category", data_type=DataType.TEXT),
                Property(name="region", data_type=DataType.TEXT),
            ],
        )
    else:
        # vectorizer none -> user supplies vectors
        vec_conf = Configure.Vectors.none()
        client.collections.create(
            name=class_name,
            vector_config=vec_conf,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source_doc", data_type=DataType.TEXT),
                Property(name="chunk_start", data_type=DataType.INT),
                Property(name="chunk_end", data_type=DataType.INT),
                Property(name="category", data_type=DataType.TEXT),
                Property(name="region", data_type=DataType.TEXT),
            ],
        )

def upsert_chunks_to_weaviate(client: weaviate.WeaviateClient, class_name: str,
                              chunks: List[Dict[str, Any]], vectors: List[List[float]] = None):
    """
    Upsert chunk objects into Weaviate. If vectors provided, pass them to batch.add_object(vector=vec).
    If vectors is None and WEAVIATE_VECTORIZE_WITH_OLLAMA is True, do not pass vectors.
    """
    col = client.collections.use(class_name)
    with col.batch.fixed_size(batch_size=200) as batch:
        for i, chunk in enumerate(chunks):
            vec = None
            if vectors:
                vec = vectors[i]
            batch.add_object(
                properties={
                    "text": chunk["text"],
                    "source_doc": chunk["source_doc"],
                    "chunk_start": int(chunk["start"]),
                    "chunk_end": int(chunk["end"]),
                    "category": chunk.get("category", "Healthcare"),
                    "region": chunk.get("region", "US"),
                },
                vector=vec  # will be None if you want Weaviate to vectorize via configured vectorizer
            )
            if batch.number_errors > 10:
                print("Too many errors in batch, aborting.")
                break

# Retrieval and RAG
def weaviate_hybrid_search(client: weaviate.WeaviateClient, class_name: str,
                           query_text: str, query_vector: List[float] = None,
                           top_k: int = TOP_K, alpha: float = HYBRID_ALPHA, filters: Any = None):
    """
    Hybrid search using Weaviate's collections.query.hybrid or collections.query.near_text depending on config.
    If collection is configured with ollama vectorizer and generative config, may also use .generate.near_text
    for a RAG generation handled by Weaviate.
    """
    col = client.collections.use(class_name)
    # If hybrid query (text + vector)
    if query_vector is not None:
        res = col.query.hybrid(query=query_text, vector=query_vector, alpha=alpha, limit=top_k,
                               return_properties=["text", "source_doc", "chunk_start", "chunk_end", "category", "region"])
    else:
        # fallback to near_text (weaviate will vectorize the query text server-side)
        res = col.query.near_text(query=query_text, limit=top_k,
                                  return_properties=["text", "source_doc", "chunk_start", "chunk_end", "category", "region"])
    hits = [obj.properties.get("text") for obj in res.objects]
    return hits

def weaviate_generate_with_context(client: weaviate.WeaviateClient, class_name: str, query_text: str, top_k: int = TOP_K):
    """
    Use Weaviate's built-in generative RAG: collections.generate.near_text(...)
    It will retrieve and then call the configured generative model in Weaviate (e.g. Ollama).
    Returns the generated text.
    """
    col = client.collections.use(class_name)
    resp = col.generate.near_text(query=query_text, limit=top_k, grouped_task="Answer user query using these facts.")
    # generative response available at resp.generative.text
    return getattr(resp.generative, "text", "")

# LLM direct calls (Mistral)
def llm_generate_with_ollama(prompt: str) -> str:
    """
    Call Ollama local API /api/generate.
    """
    url = f"{WEAVIATE_OLLAMA_ENDPOINT.rstrip('/')}/api/generate"
    payload = {"model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    # Ollama returns 'response' field
    return j.get("response", "") or j.get("text", "")

def llm_generate_with_mistral_api(prompt: str) -> str:
    """
    Call Mistral remote API (if have a key).
    """
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "open-mistral-7b", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]

def llm_generate_direct(prompt: str) -> str:
    if USE_OLLAMA_LOCAL_FOR_LLM:
        return llm_generate_with_ollama(prompt)
    elif USE_MISTRAL_API:
        return llm_generate_with_mistral_api(prompt)
    else:
        raise RuntimeError("No LLM configured. Set USE_OLLAMA_LOCAL_FOR_LLM or USE_MISTRAL_API.")

# Main flow
def main():
    print("Load dataset (wiki_snippets small subset)...")
    dataset = load_dataset("wiki_snippets", "en", split="train[:500]")
    docs = [d["passage_text"] for d in dataset if d.get("passage_text")]
    if not docs:
        docs = [
            "Healthcare in the US is expensive but provides advanced medical technology.",
            "The US government offers Medicare and Medicaid as public healthcare programs.",
            "Private insurance is the primary way people access healthcare in the US."
        ]
    print(f"Loaded {len(docs)} docs (using {min(200, len(docs))} for ingest demo).")

    # Build RDF graph from first doc (demo)
    g, EX = build_rdf_graph_for_docs(docs[0])
    graph_texts = query_rdf_for_policy_texts(g, EX)
    print("RDF graph sample text:", graph_texts[:1])

    # Chunk docs
    print("Chunking documents...")
    chunks = []
    for doc_idx, doc_text in enumerate(docs[:200]):  # limit to first 200 docs for demo
        doc_chunks = chunk_text_by_words(doc_text, max_words=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP_WORDS)
        for chunk_text, start, end in doc_chunks:
            chunks.append({
                "text": chunk_text,
                "source_doc": f"doc{doc_idx}",
                "start": start,
                "end": end,
                "category": "Healthcare",
                "region": "US"
            })
    print(f"Prepared {len(chunks)} chunks.")

    # Connect to Weaviate
    print("Connecting to Weaviate...")
    client = connect_weaviate_client()
    print("Weaviate ready:", client.is_ready())

    # Create collection (class) and configure vectorizer/generative if desired
    print("Creating Weaviate collection/class...")
    create_weaviate_collection(client, WEAVIATE_CLASS)
    time.sleep(0.5)

    # If WEAVIATE_VECTORIZE_WITH_OLLAMA is False: compute vectors locally and pass them when upserting
    vectors = None
    if not WEAVIATE_VECTORIZE_WITH_OLLAMA:
        print("Computing local embeddings for chunks (fallback)...")
        texts_for_embedding = [c["text"] for c in chunks]
        vectors = embed_texts(texts_for_embedding)  # may call Ollama embeddings or local sentence-transformers
        print("Embeddings computed.")

    # Upsert into Weaviate (Weaviate will vectorize server-side if configured)
    print("Upserting chunks into Weaviate (batch)...")
    upsert_chunks_to_weaviate(client, WEAVIATE_CLASS, chunks, vectors=vectors)
    print("Upsert finished.")

    # Prepare a user query and embed it (if desired)
    user_query = "What are the healthcare benefits in the US?"
    print("\nUser query:", user_query)

    # If you want to use a client-side vector for hybrid search, compute it here (embedding or local)
    try:
        query_vec = embed_texts([user_query])[0]
    except Exception as e:
        print("Query embedding failed; will rely on server-side vectorization:", e)
        query_vec = None

    # Hybrid retrieval
    print("Running hybrid search on Weaviate (text+vector)...")
    top_docs = weaviate_hybrid_search(client, WEAVIATE_CLASS, query_text=user_query, query_vector=query_vec,
                                     top_k=TOP_K, alpha=HYBRID_ALPHA)
    print("Top retrieved documents (snippets):")
    for i, t in enumerate(top_docs, 1):
        print(f"{i}. {t[:200]}...")

    # Try Weaviate's generative RAG (if configured generative integration)
    print("\nAttempting Weaviate generative RAG (server-side) if configured...")
    try:
        gen_answer = weaviate_generate_with_context(client, WEAVIATE_CLASS, user_query, top_k=TOP_K)
        if gen_answer:
            print("\n-- Generative answer from Weaviate (server-side) --")
            print(gen_answer)
    except Exception as e:
        print("Weaviate generative RAG not available or failed:", e)

    # Compose final prompt combining retrieved docs and RDF text
    context = "\n".join(top_docs + graph_texts[:1])
    final_prompt = f"Answer based on the context below:\n\n{context}\n\nQ: {user_query}\nA:"

    print("\nComposed prompt (truncated):")
    print(final_prompt[:900] + ("\n... [truncated]" if len(final_prompt) > 900 else ""))

    # Generate with LLM (ollama local or mistral remote)
    print("\nCalling LLM for final answer...")
    answer = llm_generate_direct(final_prompt)
    print("\n=== Final Answer ===")
    print(answer)

    # Clean up
    client.close()
    print("Done.")

if __name__ == "__main__":
    main()