import os
import ftfy
from typing import List
import pathway as pw  # <--- NEW: Pathway Import
from langchain_core.documents import Document

class UniversalTextLoader:
    """
    A robust loader that uses PATHWAY for ingestion but returns standard LangChain Documents.
    """
    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def load(self) -> List[Document]:
        documents = []
        
        print(f"[Loader] Scanning '{self.directory_path}' using Pathway...")

        # ---------------------------------------------------------
        # PATHWAY INTEGRATION
        # ---------------------------------------------------------
        # 1. Use Pathway to scan directory and read files as binary
        # mode="static" = Batch mode (reads once and stops)
        t = pw.io.fs.read(
            self.directory_path,
            format="binary",
            mode="static",
            with_metadata=True
        )

        # 2. Materialize to Pandas (The Bridge)
        # This executes the Pathway pipeline and brings data into Python memory
        df = pw.debug.table_to_pandas(t)
        
        print(f"[Loader] Pathway ingested {len(df)} files.")

        # ---------------------------------------------------------
        # PROCESSING LOOP
        # ---------------------------------------------------------
        for _, row in df.iterrows():
            # Pathway metadata is stored in a struct/dict
            metadata = row.get('_metadata', {})
            path = metadata.get('path', 'unknown.txt')
            
            # Filter: Only process .txt files
            if not str(path).endswith(".txt"):
                continue

            # Get raw bytes
            file_bytes = row['data']
            
            # Decode using robust fallback logic
            content = self._decode_bytes_safe(file_bytes)
            
            if content:
                # Fix encoding artifacts
                cleaned_content = ftfy.fix_text(content)
                
                # Create LangChain Document
                filename = os.path.basename(path)
                doc = Document(
                    page_content=cleaned_content,
                    metadata={"source": filename}
                )
                documents.append(doc)
                print(f"   -> Successfully loaded: {filename} ({len(cleaned_content)} chars)")
            else:
                print(f"   [!] Failed to decode: {os.path.basename(path)}")

        return documents

    def _decode_bytes_safe(self, data: bytes) -> str:
        """
        Tries multiple encodings to decode raw bytes.
        """
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        
        for enc in encodings:
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return None

# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    # Assumes data is in "data/" folder in the project root
    loader = UniversalTextLoader("./data") 
    docs = loader.load()
    
    if docs:
        print("\n--- Test Success! ---")
        print(f"First Book: {docs[0].metadata['source']}")
        print(f"Excerpt: {docs[0].page_content[:200]}...")
        print(f"Type: {type(docs[0])}") # Should be <class 'langchain_core.documents.base.Document'>