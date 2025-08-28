# https://docs.python-arango.com/en/main/index.html
# pip install python-arango --upgrade

from arango import ArangoClient

# Initialize the ArangoDB client.
client = ArangoClient(hosts='http://localhost:8529')

# Connect to "_system" database as root user.
# This returns an API wrapper for "_system" database.
sys_db = client.db('_system', username='root', password='passwd')

# Create a new database named "test_db" if it does not exist.
if not sys_db.has_database('test_db'):
    sys_db.create_database('test_db')

# Connect to "test_db" database as root user.
# This returns an API wrapper for "test_db" database.
db = client.db('test_db', username='root', password='passwd')

# Create collection (document)
if not db.has_collection('persons'):
    db.create_collection('persons')

# Insert document
persons = db.collection('persons')
persons.insert({'_key': 'alice', 'name': 'Alice'})
persons.insert({'_key': 'bob', 'name': 'Bob'})

# Create graph
if not db.has_graph('social'):
    graph = db.create_graph('social')
    graph.create_vertex_collection('persons')
    graph.create_edge_definition(
        edge_collection='relations',
        from_vertex_collections=['persons'],
        to_vertex_collections=['persons']
    )

# Insert edge (Alice -> Bob)
relations = db.collection('relations')
relations.insert({'_from': 'persons/alice', '_to': 'persons/bob', 'type': 'FRIENDS_WITH'})

# Query by AQL
cursor = db.aql.execute('FOR v, e, p IN 1..1 OUTBOUND "persons/alice" relations RETURN v')
for doc in cursor:
    print(doc)
