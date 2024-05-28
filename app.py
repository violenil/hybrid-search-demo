from flask import Flask, request, jsonify, render_template
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
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-de', trust_remote_code=True) # trust_remote_code is needed to use the encode method
index = faiss.read_index('index.faiss')
with open('documents.pkl', 'rb') as f:
    documents = pickle.load(f)

ix = open_dir("whoosh_index")

def bm25_search(query):
    qp = QueryParser("content", schema=ix.schema)
    q = qp.parse(query)
    with ix.searcher() as searcher:
        results = searcher.search(q, limit=5)
        return [dict(text=result['content']) for result in results]

def hybrid_search(query):
    # BM25 part
    bm25_results = bm25_search(query)
    bm25_texts = [result['text'] for result in bm25_results]
    
    # Embedding part
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=5)
    embedding_results = [documents[idx] for idx in indices[0]]

    # Combine results (simple example: just concatenating, can be improved)
    combined_results = bm25_texts + embedding_results
    return [{'text': text} for text in combined_results]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    bm25_results = bm25_search(query)
    hybrid_results = hybrid_search(query)
    return jsonify(bm25=bm25_results, hybrid=hybrid_results)

if __name__ == '__main__':
    app.run(debug=True)

