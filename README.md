# AI-Scalable-GenAI-n-Multi-Agent

## Protégé
the official website: https://protege.stanford.edu

## Neo4j
Download from: https://neo4j.com/download
Install 'Neosemantics' plugin

## Convert CSV to RDF Using OpenRefine
Download OpenRefine: https://openrefine.org/download.html

Download RDF plugin: https://github.com/stkenny/grefine-rdf-extension

```bash
products.csv:
id,name,category,price
P001,Laptop,Electronics,1200
P002,Chair,Furniture,200
P003,Smartphone,Electronics,800
```

Export RDF as Turtle
```bash
Save as products.ttl
```

## Tooling n frameworks for Graph augmented LLMs
```bash
LangChain: integrates vector + graph retrieval with LLms chains
LangGraph: agent workflows with memory n tool calls
Neo4j
Haystack: RAG pipelines with structured retrieval layers
# pip install haystack-ai 

```

## Weaviate local
```bash
https://docs.weaviate.io/weaviate/quickstart/local

https://docs.weaviate.io/weaviate/model-providers/ollama/generative

https://docs.weaviate.io/weaviate/model-providers/ollama/embeddings

```

```bash
docker run -d --name weaviate \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=20 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  semitechnologies/weaviate:1.25.9
```

## Ollama (LLM & Embeddings local)
```python
# LLM local
ollama pull mistral
# or:
ollama pull llama3.2

# Embedding model local
ollama pull nomic-embed-text

```

## Milvus
https://github.com/milvus-io/milvus
```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.4.0-rc.1-20240528 \
  standalone
```