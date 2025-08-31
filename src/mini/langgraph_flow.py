# pip install -U langgraph
# pip install grandalf
import matplotlib.pyplot as plt
import networkx as nx

# Define the nodes and edges based on the build_app function
nodes = ["planner", "retrieve", "summarize", "END"]
edges = [("planner", "retrieve"), ("retrieve", "summarize"), ("summarize", "END")]

# Create directed graph
G = nx.DiGraph() # site-packages\networkx\classes\__init__.py
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Draw graph
plt.figure(figsize=(8, 5))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrowsize=20)
plt.title("LangGraph Flow: Planner => Retriever => Summarizer => END", fontsize=12)
plt.show()
