import os
import sys

# --- Ensure imports work regardless of working directory ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config
from pipeline.embeddings.local_embedding import embed


def retrieve_with_threshold(collection, query: str, n_initial=10, similarity_threshold=None, metadata_filter=None):
    """
    Fetch top `n` chunks using the configured embeddings, and filter out any chunks
    with a cosine similarity below `similarity_threshold`.

    Args:
        collection: The ChromaDB collection object to query against.
        query (str): The search string.
        n_initial (int): The number of initial top results to fetch.
        similarity_threshold (float): Minimum cosine similarity to keep a result.
        metadata_filter (dict): An optional dictionary for where filtering (e.g. {"subject": "XYZ"}).

    Returns:
        dict: A dictionary mimicking ChromaDB's return struct containing ONLY the filtered chunks:
              {"documents": [...], "metadatas": [...], "distances": [...]}
    """
    if similarity_threshold is None:
        similarity_threshold = config.SIMILARITY_THRESHOLD
    
    # Generate embedding for the query
    query_emb = embed([query])[0]
    
    # Query the collection
    query_params = {
        "query_embeddings": [query_emb],
        "n_results": n_initial,
        "include": ["documents", "metadatas", "distances"]
    }
    
    if metadata_filter:
        query_params["where"] = metadata_filter

    results = collection.query(**query_params)
    
    # In ChromaDB with cosine space, distance = 1.0 - similarity
    # similarity > 0.3 means distance < 0.7
    max_distance = 1.0 - similarity_threshold
    
    filtered_docs = []
    filtered_metas = []
    filtered_dists = []
    
    if results and results.get("documents") and len(results["documents"][0]) > 0:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        
        for doc, meta, dist in zip(docs, metas, dists):
            if dist <= max_distance:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_dists.append(dist)
                
    # Return in the same structure ChromaDB uses, just filtered
    return {
        "documents": [filtered_docs] if filtered_docs else [[]],
        "metadatas": [filtered_metas] if filtered_metas else [[]],
        "distances": [filtered_dists] if filtered_dists else [[]]
    }
