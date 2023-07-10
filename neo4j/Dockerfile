# Use an official Neo4j runtime as a parent image
FROM neo4j:5.8.0

# Enable the APOC and GDS plugins
ENV NEO4J_apoc_export_file_enabled=true \
    NEO4J_apoc_import_file_enabled=true \
    NEO4J_apoc_import_file_use__neo4j__config=true \
    NEO4JLABS_PLUGINS='["apoc", "graph-data-science"]'

# Expose ports for Bolt and HTTP
EXPOSE 7687 7474
