# Use an official Elasticsearch runtime as a parent image
FROM docker.elastic.co/elasticsearch/elasticsearch:7.15.0

# The default "elasticsearch" user is used by the base image to run Elasticsearch
USER elasticsearch

# Expose ports for Elasticsearch (HTTP: 9200 and Transport: 9300)
EXPOSE 9200 9300
