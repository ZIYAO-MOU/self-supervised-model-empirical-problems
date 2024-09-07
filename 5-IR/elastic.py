from elasticsearch import Elasticsearch
from datasets import load_dataset
import json

corpus = load_dataset("scifact", "corpus")
claims = load_dataset("scifact", "claims")

es = Elasticsearch("http://localhost:9200")

# Index the corpus
for i, doc in enumerate(corpus['train']):
    abstract = doc['abstract']
    es.index(index='sifact_abstracts', id=doc['doc_id'], body={'text': abstract})

def search_claim(claim):
    response = es.search(index='sifact_abstracts', body={
        "query": {
            "match": {
                "text": claim,
            }
        },
    })
    return response['hits']['hits']

def precision_at_k(relevant_docs, retrieved_docs, k):
    retrieved_set = set(doc['_id'] for doc in retrieved_docs[:k])
    relevant_set = set(str(num) for num in relevant_docs)
    return len(retrieved_set & relevant_set) / k

def mean_precision_at_k(claims, k=5):
    precisions = []
    for claim in claims:
        print(claim)
        claim_text = claim['claim']
        relevant_doc_ids = claim['cited_doc_ids']
        retrieved_docs = search_claim(claim_text)
        precision = precision_at_k(relevant_doc_ids, retrieved_docs, k)
        precisions.append(precision)
    return sum(precisions) / len(precisions) if precisions else 0

# Calculate MP@k
k = 5  # Adjust k as needed
mp_at_k = mean_precision_at_k(claims['train'], k)
print(f"Mean Precision at {k}: {mp_at_k}")

def average_precision(relevant_docs, retrieved_docs):
    relevant_set = set(str(num) for num in relevant_docs)
    hits = 0
    sum_precision = 0.0
    for i, doc in enumerate(retrieved_docs):
        if doc['_id'] in relevant_set:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant_set) if relevant_set else 0

def mean_average_precision(claims):
    aps = []
    for claim in claims:
        claim_text = claim['claim']
        relevant_doc_ids = claim['cited_doc_ids']
        retrieved_docs = search_claim(claim_text)
        ap = average_precision(relevant_doc_ids, retrieved_docs)
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0

# Calculate MAP
map_score = mean_average_precision(claims["train"])
print(f"Mean Average Precision (MAP): {map_score}")