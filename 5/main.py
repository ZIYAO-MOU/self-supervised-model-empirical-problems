import pickle
import faiss
import numpy as np
from datasets import load_dataset

corpus = load_dataset("scifact", "corpus")
claims = load_dataset("scifact", "claims")

ground_truth = {}
for example in claims['train']:
    claim_id = example['id'] 
    relevant_docs = set(example['cited_doc_ids'])
    ground_truth[claim_id] = relevant_docs


with open('scifact_evidence_embeddings.pkl', 'rb') as f:
    openai_embeddings = pickle.load(f)

with open('scifact_claim_embeddings.pkl', 'rb') as f:
    claim_embeddings = pickle.load(f)

openai_ids = list(openai_embeddings.keys())
openai_vectors = np.array([embedding for embedding in openai_embeddings.values()])

dimension = openai_vectors.shape[1]
print(dimension)
index = faiss.IndexFlatL2(dimension)
index.add(openai_vectors) 

def search(query_embedding, k=5):
    distances, indices = index.search(np.array([query_embedding]), k)
    return distances, indices

def compute_mrr_and_map(openai_ids, claim_embeddings, ground_truth, k=5):
    mrr_total = 0.0
    map_total = 0.0
    num_queries = len(claim_embeddings)
    # openai_embeddings = openai_embeddings.items() 
    
    for claim, claim_embedding in claim_embeddings.items():
        claim_id, claim_text = claim
        claim_vector = np.array([claim_embedding])
        distances, indices = index.search(claim_vector, k)
        retrieved_evidence_ids = [openai_ids[idx][0] for idx in indices[0]]
        print(retrieved_evidence_ids)
        # Ground truth
        correct_answers = ground_truth.get(claim_id, set())

        # MRR
        reciprocal_rank = 0.0
        for rank, idx in enumerate(retrieved_evidence_ids, start=1):
            if idx in correct_answers:
                reciprocal_rank = 1.0 / rank
                break
        mrr_total += reciprocal_rank

        # AP
        relevant_found = 0
        avg_precision = 0.0
        for rank, idx in enumerate(retrieved_evidence_ids, start=1):
            if idx in correct_answers:
                relevant_found += 1
                avg_precision += relevant_found / rank
        if len(correct_answers) > 0:
            avg_precision /= len(correct_answers)
        
        map_total += avg_precision

    mrr = mrr_total / num_queries
    map_score = map_total / num_queries

    return mrr, map_score

mrr, map_score = compute_mrr_and_map(openai_embeddings, claim_embeddings, ground_truth, k=5)
print(f"MRR: {mrr:.4f}, MAP: {map_score:.4f}")

