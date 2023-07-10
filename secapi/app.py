import configparser
from elasticsearch import Elasticsearch
from neo4j import GraphDatabase, basic_auth
from sec_api import ExtractorApi
import requests
import pandas as pd
import time
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx




def get_filings(api_key, ticker):
    """
    Function to make API request for SEC filings.
    """
    base_url = "https://api.sec-api.io?token=" + api_key
    headers = {"Content-Type": "application/json"}

    query = {
      "query": {
        "query_string": {
          "query": f"filedAt:{{2022-01-01 TO 2022-12-31}} AND formType:\"10-K\" AND ticker:\"{ticker}\""
        }
      },
      "size": "200",
      "sort": [{ "filedAt": { "order": "desc" } }]
    }

    response = requests.post(base_url, headers=headers, json=query)
    return response.json() if response.status_code == 200 else None

def get_all_filings(api_key, tickers):
    """
    Function to get all filings for the given tickers.
    """
    dataframes = []  # Create an empty list to store dataframes

    for ticker in tickers:
        response = get_filings(api_key, ticker)
        if response:
            filings = response['filings']
            if len(filings) > 0:
                data = pd.DataFrame(filings)
                dataframes.append(data)  # Append the dataframe to the list
            else:
                print(f"No filings found for ticker {ticker}.")
        else:
            print("Request failed. Check API key and internet connection.")

    df_sp500 = pd.concat(dataframes, ignore_index=True)  # Concatenate all the dataframes in the list

    return df_sp500

def get_business_description(df, api_key, extractorApi):
    """
    Function to get business description from the 10-K filings.
    """
    result_df = pd.DataFrame(columns=['ticker', 'description'])

    for idx, row in df.iterrows():
        ticker = row['ticker']
        url_10k = row['linkToTxt']
        try:
            description_text = extractorApi.get_section(url_10k, "1", "text")
            new_row = pd.DataFrame({'ticker': [ticker], 'description': [description_text]})
            result_df = pd.concat([result_df, new_row], ignore_index=True)
        except Exception as e:
            print(f"Error while fetching description for {ticker}: {str(e)}")

    return result_df


def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters
    text = re.sub(r'\W', ' ', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back into a single string
    text = ' '.join(words)

    return text

def get_avg_word_vector(words, model, num_features):
    # Initialize a vector of zeros
    feature_vector = np.zeros((num_features,), dtype="float32")

    nwords = 0

    # List containing names of words in the vocabulary
    index2word_set = set(model.wv.index_to_key)

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            feature_vector = np.add(feature_vector, model.wv[word])

    if (nwords > 0):
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector

def create_minimum_spanning_tree(similarity_matrix, result):
    distance_matrix = 1 - similarity_matrix

    # Compute the minimal spanning tree
    sparse_tree = minimum_spanning_tree(distance_matrix)

    # Convert the sparse tree to a dense tree
    dense_tree = sparse_tree.toarray()

    # Create a mapping from node indices to tickers
    index_to_ticker = {i: ticker for i, ticker in enumerate(result['ticker'])}

    # Create a graph from the dense tree
    G = nx.relabel_nodes(nx.DiGraph(dense_tree), index_to_ticker)

    return G

def main():
    # Read the configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Get the API key
    api_key = config['SEC']['API_KEY']

    # Initialize the ExtractorApi with the API key
    extractorApi = ExtractorApi(api_key=api_key)

    # Gather data from SEC API
    tickers = ['A', 'AAL', 'AAP', 'AAPL', 'ZION', 'ZTS']
    df_sp500 = get_all_filings(api_key, tickers)
    result = get_business_description(df_sp500, api_key, extractorApi)

    print("RESULT: /n",result)

     # Check if the descriptions are empty
    if result['description'].apply(lambda x: len(x) == 0).all():
        print("All descriptions are empty. Cannot train Word2Vec model.")
        return

    # Preprocess the text
    result['description'] = result['description'].apply(preprocess_text)

    # Tokenize the descriptions
    result['description'] = result['description'].apply(word_tokenize)

    # Train a Word2Vec model on the descriptions
    model = Word2Vec(result['description'], min_count=1)

    # Get the average word vector for each description
    result['avg_word_vector'] = result['description'].apply(lambda x: get_avg_word_vector(x, model, model.vector_size))

    # Compute the cosine similarity between each pair of descriptions
    similarity_matrix = cosine_similarity(list(result['avg_word_vector']))

    # Print the similarity matrix
    print(similarity_matrix)

    # Create the minimum spanning tree
    G = create_minimum_spanning_tree(similarity_matrix, result)

    # Wait for Elasticsearch and Neo4j to start
    time.sleep(30)  # Adjust this value based on how long it takes for Elasticsearch and Neo4j to start

    # Connect to Elasticsearch
    es = Elasticsearch([{'host': 'elasticsearch', 'port': 9200, 'scheme': 'http'}])

    # Connect to Neo4j
    neo4j_driver = GraphDatabase.driver("bolt://neo4j:7687", auth=basic_auth("neo4j", "password"))  # Replace with your Neo4j username and password

    # Write the full text for each ticker/company into Elasticsearch
    for i, row in result.iterrows():
        doc = {
            'ticker': row['ticker'],
            'description': row['description'],
        }
        es.index(index="tickers", id=i, body=doc)

    # Write the nodes and edges into Neo4j
    with neo4j_driver.session() as session:
        for node in G.nodes:
            session.run("CREATE (a:Company {name: $name})", name=node)
        for edge in G.edges:
            session.run("""
                MATCH (a:Company),(b:Company)
                WHERE a.name = $name1 AND b.name = $name2
                CREATE (a)-[r:SIMILAR_TO]->(b)
                """, name1=edge[0], name2=edge[1])

if __name__ == "__main__":
    main()
