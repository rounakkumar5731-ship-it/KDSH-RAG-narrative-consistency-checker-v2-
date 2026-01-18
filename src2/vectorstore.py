import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any

class FaissVectorStore:
    """
    A persistent Vector Store using FAISS for search and Pickle for metadata.
    It saves two files:
    1. index_name.faiss (The mathematical vectors)
    2. index_name.pkl   (The original text chunks and IDs)
    """
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.index = None
        self.metadata: List[Dict] = []
        
        # Create directory if it doesn't exist
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)

    def add_vectors(self, vectors: np.ndarray, metadatas: List[Dict]):
        """
        Adds vectors and their corresponding text metadata to the index.
        """
        if len(vectors) == 0:
            return

        dim = vectors.shape[1]
        
        # Initialize FAISS index if not exists (L2 = Euclidean Distance)
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        
        # Add to FAISS
        self.index.add(vectors.astype('float32'))
        
        # Store metadata (contains original text, chunk_id, source)
        self.metadata.extend(metadatas)
        
        print(f"[VectorStore] Added {len(vectors)} chunks to index at '{self.persist_dir}'")

    def save(self):
        """
        Saves the FAISS index and metadata to disk.
        """
        if self.index is None:
            print("[VectorStore] Nothing to save (index is empty).")
            return

        # Paths
        faiss_path = os.path.join(self.persist_dir, "index.faiss")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        # Write Files
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
            
        print(f"[VectorStore] Index saved successfully to {self.persist_dir}")

    def load(self):
        """
        Loads the FAISS index and metadata from disk.
        """
        faiss_path = os.path.join(self.persist_dir, "index.faiss")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
            print(f"[VectorStore] No existing index found at {self.persist_dir}")
            return False

        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
            
        print(f"[VectorStore] Loaded index with {self.index.ntotal} vectors from {self.persist_dir}")
        return True

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """
        Searches the index using a vector and returns the ORIGINAL text chunks.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # FAISS search returns Distances (D) and Indices (I)
        D, I = self.index.search(query_vector.astype('float32'), top_k)
        
        results = []
        # I[0] contains the indices of the neighbors for the first query vector
        for idx, distance in zip(I[0], D[0]):
            if idx == -1: continue # No neighbor found
            
            # Retrieve the original text chunk using the index
            chunk_data = self.metadata[idx]
            
            results.append({
                "chunk_id": chunk_data.get('chunk_id'),
                "source": chunk_data.get('source'),
                "text": chunk_data.get('text'), # <--- The Original Text
                "distance": float(distance)
            })
            
        return results