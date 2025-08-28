# pip install rdflib
# pip install neo4j
from rdflib import Graph, Literal, RDF, Namespace, URIRef
import json

g = Graph()
EX = Namespace("http://example.org/employees#")
g.bind("ex", EX)
with open("src/data/employees.json") as f:
    data = json.load(f)
for item in data:
    emp_uri = URIRef(EX + item["id"])
    g.add((emp_uri, RDF.type, EX.Employee))
    g.add((emp_uri, EX.name, Literal(item["name"])))
    g.add((emp_uri, EX.role, Literal(item["role"])))
    g.add((emp_uri, EX.department, Literal(item["department"])))
    g.serialize("src/data/employees.ttl", format="turtle")
    g.serialize("src/data/employees.rdf", format="xml")

# https://neo4j.com/docs/getting-started/data-modeling/tutorial-data-modeling/
# Open Neo4j http://localhost:7687
# Create the n10s configuration: CALL n10s.graphconfig.init()
# Load RDF Data: CALL n10s.rdf.import.fetch("file:///src/data/employees.ttl", "Turtle")
# Query the Graph Using Cypher: MATCH (e:Employee) RETURN e.name, e.role, e.department
# Visualize the Graph: CALL apoc.graph.fromRDF("file:///src/data/employees.rdf")
# Clean up the Graph: MATCH (n) DETACH DELETE n

# Query with Cypher
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    result = session.run("MATCH (e:Employee) RETURN e.name, e.role, e.department")
    for record in result:
        print(record["e.name"], record["e.role"], record["e.department"])
driver.close()

# Query with SPARQL (optional)
from rdflib.plugins.sparql import prepareQuery

query = prepareQuery(
    """
    SELECT ?name ?role ?department
    WHERE {
        ?e a ex:Employee .
        ?e ex:name ?name .
        ?e ex:role ?role .
        ?e ex:department ?department .
    }
    """,
    initNs={"ex": EX}
)

for row in g.query(query):
    print(f"Name: {row.name}, Role: {row.role}, Department: {row.department}")

# Results
'''
Name: Alice, Role: Engineer, Department: R&D
Name: Bob, Role: Manager, Department: HR
'''