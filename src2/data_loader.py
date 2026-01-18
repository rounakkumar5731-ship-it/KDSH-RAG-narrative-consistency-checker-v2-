import os
import glob
from typing import List
import ftfy
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

class UniversalTextLoader:
    """
    A robust loader for .txt files that handles:
    1. Multiple encoding attempts (UTF-8, Latin-1, etc.)
    2. Text cleaning using ftfy (fixes bad characters)
    3. LangChain Document format compatibility
    """
    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def load(self) -> List[Document]:
        documents = []
        # Find all .txt files recursively
        search_path = os.path.join(self.directory_path, "**/*.txt")
        file_paths = glob.glob(search_path, recursive=True)
        
        print(f"[Loader] Scanning '{self.directory_path}'...")
        print(f"[Loader] Found {len(file_paths)} text files.")

        for file_path in file_paths:
            content = self._read_file_safe(file_path)
            if content:
                # Fix encoding artifacts (The "screenshot" error fix)
                cleaned_content = ftfy.fix_text(content)
                
                # Create LangChain Document
                filename = os.path.basename(file_path)
                doc = Document(
                    page_content=cleaned_content,
                    metadata={"source": filename}
                )
                documents.append(doc)
                print(f"   -> Successfully loaded: {filename} ({len(cleaned_content)} chars)")
            else:
                print(f"   [!] Failed to read: {os.path.basename(file_path)}")

        return documents

    def _read_file_safe(self, file_path: str) -> str:
        """
        Tries multiple encodings to read the file.
        """
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        return None

# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    # Assumes data is in "data/" folder in the project root
    # Adjust path as necessary
    loader = UniversalTextLoader("./data") 
    docs = loader.load()
    
    if docs:
        print("\n--- Test Success! ---")
        print(f"First Book: {docs[0].metadata['source']}")
        print(f"Excerpt: {docs[0].page_content[:200]}...")
        print(type(docs[0]))
        print(type(docs))