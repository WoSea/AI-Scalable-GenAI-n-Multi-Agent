# pip install openai faiss-cpu rdflib pinecone-client langchain datasets tqdm python-dotenv
# MacOS/Linux: export OPENAI_API_KEY="your_api_key_here"
# Windows PowerShell: setx OPENAI_API_KEY "your_api_key_here"   #$env:OPENAI_API_KEY="your_api_key_here"
# Use Mistral or Llama2 or GPT-4All instead of OpenAI for local LLM inference
from datasets import load_dataset
from rdflib import Graph, URIRef, Literal, Namespace, RDF
from openai import OpenAI
import openai
import numpy as np
import faiss
import pinecone

'''
This is sample results
'''

dataset = load_dataset("wiki_snippets", "en", split="train[:500]") # smaller subset
docs = [d["passage_text"] for d in dataset if d["passage_text"]]

'''
docs = [
  "Healthcare in the US is expensive but provides advanced medical technology.",
  "The US government offers Medicare and Medicaid as public healthcare programs.",
  "Private insurance is the primary way people access healthcare in the US."
]
'''

# =========================== Build RDF Knowledge Graph (please ref json_to_rdf.py file)
g = Graph()
EX = Namespace("http://example.org/")
g.add((EX["Doc1"], RDF.type, EX["Policy"]))
g.add((EX["Doc1"], EX["category"], Literal("Healthcare")))
g.add((EX["Doc1"], EX["region"], Literal("US")))
g.add((EX["Doc1"], EX["text"], Literal(docs[0])))
g.serialize(destination="src/data/knowledge.ttl", format="turtle") # save to disk

# =========================== Embed Documents
client = OpenAI()
openai.api_key = 'api_key'

response = client.embeddings.create(
    input=docs,  # docs, string
    model="text-embedding-3-small"
)
embeddings = [item.embedding for item in response.data] # list of vectors
'''
embeddings[0] = [0.01, -0.02, 0.13, ...]   # 1536 dimension
embeddings[1] = [0.04, -0.01, 0.09, ...]
embeddings[2] = [0.00,  0.03, 0.12, ...]
'''

# =========================== Create FAISS Index: Embedding entered Faiss for quick query on Local
dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32")) # add vectors to index to enable similarity search (nearest neighbors)

# =========================== Metadata-Aware Retrieval with Pinecone (Can be combine Milvus + Postgres/MySQL)
pinecone.init(api_key="pinecone_key", environment="us-east-1")
index_name = "rag-hybrid-demo"
pinecone.create_index(index_name, dimension=1536, metric="cosine",
metadata_config={"indexed": ["category", "region"]})
pc_index = pinecone.Index(index_name)

pc_index.upsert([  # Upsert Vectors with Metadata
{
    "id": f"doc{i}",
    "values": embeddings[i],
    "metadata": {"category": "Healthcare", "region": "US", "text": docs[i]}
}
for i in range(len(docs[:100]))
])

'''
{
  "id": "doc1",
  "values": [0.04, -0.01, 0.09, ...],
  "metadata": {
    "category": "Healthcare",
    "region": "US",
    "text": "The US government offers Medicare and Medicaid as public healthcare programs."
  }
}
'''

# =========================== Querying the Index
query = "What are the healthcare benefits in the US?"
query_response = client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
)
query_embedding = query_response.data[0].embedding

'''
query_embedding = [0.02, 0.01, 0.15, ...]   # vector 1536 dimension
'''

# =========================== Vector Search FAISS
D, I = index.search(np.array([query_embedding]).astype('float32'), k=3)
top_faiss_docs = [docs[i] for i in I[0]]

'''
I = [[1, 2]]
D = [[0.12, 0.19]]
top_faiss_docs = [
  "The US government offers Medicare and Medicaid as public healthcare programs.",
  "Private insurance is the primary way people access healthcare in the US."
]
'''

# =========================== Pinecone Filtered Search
pinecone_results = pc_index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True,
    filter={"category": {"$eq": "Healthcare"}, "region": {"$eq": "US"}}
) 
top_pinecone_docs = [match["metadata"]["text"] for match in
pinecone_results["matches"]]

'''
top_pinecone_docs = [
  "Healthcare in the US is expensive but provides advanced medical technology.",
  "The US government offers Medicare and Medicaid as public healthcare programs."
]
'''

# =========================== Knowledge Graph Filtering via SPARQL
query = """
PREFIX ex: <http://example.org/>
SELECT ?text WHERE {
  ?doc a ex:Policy ;
       ex:category "Healthcare" ;
       ex:region "US" ;
       ex:text ?text .
}
"""
results = g.query(query)
graph_texts = []
for row in results:
    graph_texts.append(str(row.text))

'''
graph_texts = ["Healthcare in the US is expensive but provides advanced medical technology."]
'''

# =========================== Compose Final Prompt to LLM
context = "\n".join(top_faiss_docs + top_pinecone_docs + graph_texts[:1])
final_prompt = f"Answer based on the context below:\n\n{context}\n\nQ: {query}\nA:"

'''
Answer based on the context below:

The US government offers Medicare and Medicaid as public healthcare programs.
Private insurance is the primary way people access healthcare in the US.
Healthcare in the US is expensive but provides advanced medical technology.

Q: What are the healthcare benefits in the US?
A:

'''

# =========================== Generate Response with LLM (OpenAI)/Mistral
response = openai.ChatCompletion.create(
    model="gpt-5",
    messages=[{"role": "user", "content": final_prompt}]
) 
print(response["choices"][0]["message"]["content"])

'''
In the US, healthcare benefits include access to advanced medical technology, public programs such as Medicare and Medicaid, and private insurance options. However, costs remain high compared to many other countries.
'''