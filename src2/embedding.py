import time
from typing import List, Dict
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

class EmbeddingPipeline:
    """
    Handles splitting text into chunks, tagging them with IDs, and generating vectors.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        print(f"[Embedder] Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents and assigns a unique 'chunk_id' to each chunk
        relative to its source book.
        """
        if not documents:
            print("[Embedder] Warning: No documents provided to chunk.")
            return []
            
        print(f"[Embedder] Splitting {len(documents)} documents...")
        
        # 1. Perform the split
        raw_chunks = self.text_splitter.split_documents(documents)
        
        # 2. Assign Chunk IDs (Track separate counters for each book)
        chunk_counters: Dict[str, int] = {}
        processed_chunks = []

        for chunk in raw_chunks:
            # Get the filename (e.g., 'The Count of Monte Cristo.txt')
            source = chunk.metadata.get('source', 'unknown')
            
            # Initialize counter for this book if new
            if source not in chunk_counters:
                chunk_counters[source] = 0
            
            # Assign ID and increment
            chunk.metadata['chunk_id'] = chunk_counters[source]
            chunk_counters[source] += 1
            
            processed_chunks.append(chunk)
        
        print(f"[Embedder] Created {len(processed_chunks)} chunks total.")
        
        # Debug: Show how many chunks per book
        for source, count in chunk_counters.items():
            print(f"   -> {source}: {count} chunks")
            
        return processed_chunks

    def embed_chunks(self, chunks: List[Document]) -> np.ndarray:
        """
        Converts text chunks into vector embeddings.
        """
        if not chunks:
            return np.array([])

        texts = [chunk.page_content for chunk in chunks]
        
        print(f"[Embedder] Generating embeddings for {len(texts)} chunks...")
        start_time = time.time()
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        duration = time.time() - start_time
        print(f"[Embedder] Finished in {duration:.2f}s. Shape: {embeddings.shape}")
        
        return embeddings

# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    from data_loader import UniversalTextLoader
    
    # 1. Load
    loader = UniversalTextLoader("../data")
    docs = loader.load()
    
    if docs:
        # 2. Chunk
        pipeline = EmbeddingPipeline()
        chunks = pipeline.chunk_documents(docs)
        
        # 3. Verify Chunk IDs
        if len(chunks) > 0:
            first_chunk = chunks[0]
            print(f"\n--- Chunk Metadata Check ---")
            print(f"Source: {first_chunk.metadata['source']}")
            print(f"Chunk ID: {first_chunk.metadata['chunk_id']}") # Should be 0
            
            # Check a later chunk
            if len(chunks) > 10:
                print(f"10th Chunk ID: {chunks[10].metadata['chunk_id']}") # Should be 10