from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel
import faiss
import numpy as np
import pickle
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = Flask(__name__)

# Load the model, index, and documents
print("LOADING MODEL")
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-de', trust_remote_code=True)
print("LOADING INDEX")
index = faiss.read_index('index.faiss')
with open('documents.pkl', 'rb') as f:
    documents = pickle.load(f)

ix = open_dir("whoosh_index")
print("EVERYTHING LOADED")
def normalize_bm25_scores(scores):
    normalized_scores = []
    for score in scores:
        score['score'] = score['score'] / (score['score'] + 10)
    return normalized_scores


def weighted_sum(bm25_scores, vector_scores, weights={'bm25': 0.4, 'vector': 0.6}):
    # Normalize BM25 scores
    normalized_bm25_scores = normalize_bm25_scores(bm25_scores)

    # Interpolation for missing scores (ensure both lists are of the same length)
    min_bm25_score = min([x['score'] for x in bm25_scores], default=0)
    min_vector_score = min([x['score'] for x in vector_scores], default=0)

    bm25_scores_dict = {result['id']: result['score'] for result in normalized_bm25_scores}
    vector_scores_dict = {result['id']: result['score'] for result in vector_scores}
    all_ids = set([result['id'] for result in bm25_scores] + [result['id'] for result in vector_scores])
    for id in all_ids:
        if id not in bm25_scores_dict:
            bm25_scores_dict[id] = min_bm25_score
        if id not in vector_scores_dict:
            vector_scores_dict[id] = min_vector_score

    # Weight and Sum
    combined_scores = []
    for id in all_ids:
        combined_score = weights['bm25'] * bm25_scores_dict[id] + weights['vector'] * vector_scores_dict[id]
        combined_scores.append({'score': combined_score, 'id': id, 'text': documents[id]})

    return combined_scores


def bm25_search(query):
    qp = QueryParser("content", schema=ix.schema)
    q = qp.parse(query)
    with ix.searcher() as searcher:
        results = searcher.search(q, limit=20)  # Adjust limit as necessary
        return [{'text': result['content'], 'score': result.score, 'id': result.docnum} for result in results]

def hybrid_search(bm25_results, vector_results):
    # Combine scores using weighted sum
    combined_scores = weighted_sum(bm25_results, vector_results) # [{'score': combined_score, 'id': doc_id}]
    # sort the combined scores in descending order using lamda
    sorted_combined_results = sorted(combined_scores, key=lambda x: x['score'], reverse=True)
    print("Combined Results:", sorted_combined_results)

    return sorted_combined_results[:20]  # Limit to top 20 results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/vector_search', methods=['POST'])
def vector_search(query, k=20):
    query_embedding = model.encode([query])
    _, indices, vectors = index.search_and_reconstruct(np.array(query_embedding), k=k)

    # Filter out invalid indices (-1) and corresponding distances
    valid_indices = [idx for idx in indices[0] if idx != -1]
    valid_vectors = [dist for dist, idx in zip(vectors[0], indices[0]) if idx != -1]
    cosine_scores = [cosine_similarity(query_embedding, [vector]) for vector in valid_vectors]
    # Return results with valid indices and distances
    results = [{'text': documents[idx], 'score': float(score), 'id': idx} for score, idx in zip(cosine_scores, valid_indices)]

    return results

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    bm25_results = bm25_search(query)
    vector_results = vector_search(query)
    hybrid_results = hybrid_search(bm25_results, vector_results)
    display_results_bm25 = [{'text': result['text'], 'score': result['score']} for result in bm25_results]
    display_results_vector = [{'text': result['text'], 'score': result['score']} for result in vector_results]
    display_results_hybrid = [{'text': result['text'], 'score': result['score']} for result in hybrid_results]

    return jsonify(bm25=display_results_bm25, vector=display_results_vector, hybrid=display_results_hybrid)

if __name__ == '__main__':
    app.run(debug=True)

