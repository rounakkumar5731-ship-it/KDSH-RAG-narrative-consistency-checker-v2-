import os
import numpy as np
from src2.data_loader import UniversalTextLoader
from src2.embedding import EmbeddingPipeline
from src2.vectorstore import FaissVectorStore

# CONFIGURATION
DATA_DIR = "./data"
STORE_DIR_MC = "./faiss_store_monte_cristo"
STORE_DIR_castaways = "./faiss_store_castaways"

# Filenames must match exactly what is in your data folder
FILE_MC = "The Count of Monte Cristo.txt"
FILE_CASTAWAYS = "In search of the castaways.txt"

def build_indices():
    """
    Step 1: Ingest, Chunk, Embed, and Save separate indices.
    """
    print("=== BUILDING INDICES ===")
    
    # 1. Load All Data
    loader = UniversalTextLoader(DATA_DIR)
    all_docs = loader.load()
    
    # 2. Initialize Pipeline
    pipeline = EmbeddingPipeline()
    
    # --- PROCESS BOOK 1: MONTE CRISTO ---
    print(f"\n--- Processing: {FILE_MC} ---")
    docs_mc = [d for d in all_docs if FILE_MC in d.metadata.get('source', '')]
    
    if docs_mc:
        chunks_mc = pipeline.chunk_documents(docs_mc)
        vectors_mc = pipeline.embed_chunks(chunks_mc)
        
        # Prepare metadata for storage
        meta_mc = [{
            "text": c.page_content,
            "chunk_id": c.metadata.get('chunk_id'),
            "source": c.metadata.get('source')
        } for c in chunks_mc]
        
        # Save to Store 1
        store_mc = FaissVectorStore(STORE_DIR_MC)
        store_mc.add_vectors(vectors_mc, meta_mc)
        store_mc.save()
    else:
        print(f"[!] Warning: {FILE_MC} not found in data.")

    # --- PROCESS BOOK 2: CASTAWAYS ---
    print(f"\n--- Processing: {FILE_CASTAWAYS} ---")
    docs_cast = [d for d in all_docs if FILE_CASTAWAYS in d.metadata.get('source', '')]
    
    if docs_cast:
        chunks_cast = pipeline.chunk_documents(docs_cast)
        vectors_cast = pipeline.embed_chunks(chunks_cast)
        
        meta_cast = [{
            "text": c.page_content,
            "chunk_id": c.metadata.get('chunk_id'),
            "source": c.metadata.get('source')
        } for c in chunks_cast]
        
        # Save to Store 2
        store_cast = FaissVectorStore(STORE_DIR_castaways)
        store_cast.add_vectors(vectors_cast, meta_cast)
        store_cast.save()
    else:
        print(f"[!] Warning: {FILE_CASTAWAYS} not found in data.")

def query_book(book_name, query_text):
    """
    Step 2: Query a specific store and get original evidence.
    """
    print(f"\n=== QUERYING: {book_name} ===")
    print(f"Query: '{query_text}'")
    
    # Select the correct store directory
    if "Monte Cristo" in book_name:
        store_path = STORE_DIR_MC
    elif "Castaways" in book_name:
        store_path = STORE_DIR_castaways
    else:
        print("Unknown book.")
        return

    # Load Store
    store = FaissVectorStore(store_path)
    if not store.load():
        return

    # Generate Query Vector (Need the embedding pipeline just for encoding)
    pipeline = EmbeddingPipeline() # Re-init pipeline for encoding query
    # Note: embed_chunks expects a list of docs, but we can access the model directly
    query_vector = pipeline.model.encode([query_text])
    
    # Search
    results = store.search(query_vector, top_k=3)
    
    # Display Results
    for i, res in enumerate(results):
        print(f"\n[Result {i+1}] (Chunk {res['chunk_id']})")
        print(f"Evidence: \"{res['text'][:200]}...\"") # Showing first 200 chars
        print(f"Distance: {res['distance']:.4f}")

if __name__ == "__main__":
    # 1. Build the separate stores (Run this once)
    build_indices()
    
    # 2. Test Query for Monte Cristo
    query_book("Monte Cristo", "Faria died in prison after a seizure")
    
    # 3. Test Query for Castaways
    query_book("Castaways", "The earthquake in the Andes")