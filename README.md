# Hybrid Search with Jina Embeddings v2

Setup a search app that performs 3 different searches:
    1. BM25
    2. Vector Search
    3. Hybrid Search (BM25 + Vector Search)

The search is performed on an index that consists of products from the German XMarket dataset (see [Huggingface XMarket Dataset](https://huggingface.co/datasets/jinaai/xmarket_de).
Those are products with titles and descriptions in German. The index contains a concatenation of these titles and descriptions as well as a vector representation 
of the text using the German-English bilingual
`jinaai/jina-embeddings-v2-base-de` embedding model from the Huggingface model hub. This makes them searchable with BM25 and Vector Search.

## Setup

Install the requirements:

```bash
pip install -r requirements.txt
```

Create the index:

```bash
python index.py
```

Run the app:

```bash
python app.py
```

Now, your app should be running locally and accessible at http://127.0.0.1:5000.

You can run a search in German as well in English:

![English_search.png](resources%2FEnglish_search.png)
![German_search.png](resources%2FGerman_search.png)