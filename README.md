# AI-Scalable-GenAI-n-Multi-Agent

## Protégé
the official website: https://protege.stanford.edu

## Neo4j
Download from: https://neo4j.com/download
Install 'Neosemantics' plugin

## Convert CSV to RDF Using OpenRefine
Download OpenRefine: https://openrefine.org/download.html

Download RDF plugin: https://github.com/stkenny/grefine-rdf-extension

```
products.csv:
id,name,category,price
P001,Laptop,Electronics,1200
P002,Chair,Furniture,200
P003,Smartphone,Electronics,800
```

Export RDF as Turtle
```
Save as products.ttl
```

## Tooling n frameworks for Graph augmented LLMs
```
LangChain: integrates vector + graph retrieval with LLms chains
LangGraph: agent workflows with memory n tool calls
Neo4j
Haystack: RAG pipelines with structured retrieval layers
# pip install haystack-ai 

```