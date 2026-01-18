
# Narrative Consistency Checker (Track A: Agentic RAG)

A Logic-Driven RAG system designed to verify character backstories against long-form novels (100k+ words) by detecting mutually exclusive timeline states.

## 📖 Overview
This project solves the "Global Consistency" problem in Generative AI. Instead of treating a book as a simple text stream, this system treats it as a **database of temporal facts**. It ingests raw text, partitions it into timeline-aware chunks, and uses an Agentic LLM to act as a "Logic Judge"—determining if a proposed backstory is legally possible within the established reality of the novel.

## 🚀 Key Features
* **Timeline-Aware RAG:** Sorts retrieved evidence chronologically (via `chunk_id`) to detect anachronisms and causal contradictions.
* **Agentic Logic Judge:** A hybrid reasoning engine that strictly checks for "Mutually Exclusive States" (e.g., *Alive vs Dead*, *Prison vs Freedom*) rather than just semantic similarity.
* **Auto-Fallback Routing:** Dynamic model switching between `Qwen-32b`, `Llama-3`, and `Mixtral` to ensure 100% uptime despite API rate limits.
* **"Innocent Until Proven Guilty":** Logic architecture that defaults to consistency unless explicit contradictory evidence is retrieved.

## 🛠️ Architecture
1. **Ingestion:** Universal loader reads and cleans raw `.txt` files (UTF-8/Latin-1 compatible).
2. **Indexing:** `FAISS` stores vector embeddings, separated by book to prevent cross-contamination.
3. **Extraction:** Regex/LLM extracts atomic facts from backstories (e.g., *"He died in 1815"*).
4. **Retrieval:** Targeted vector search retrieves the top 10 most relevant timeline chunks.
5. **Judgment:** The LLM compares facts vs. evidence to output a binary consistency score (0 or 1).

## 📦 Installation

**1. Clone the Repository**
```bash
git clone [https://github.com/YOUR_USERNAME/narrative-consistency-checker.git](https://github.com/YOUR_USERNAME/narrative-consistency-checker.git)
cd narrative-consistency-checker

```

**2. Set up Environment Variables**
Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_api_key_here

```

**3. Install Dependencies**
It is recommended to use a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
source .venv/bin/activate      # On Linux/Mac
pip install -r requirements.txt

```

## 🏃‍♂️ Usage

**Step 1: Build the Vector Indices**
Run this once to parse the books and create the FAISS database. This processes the text files in the `data/` directory.

```powershell
python build_and_query.py

```

*(Note: If you skip this, the main script will automatically run it for you on the first launch.)*

**Step 2: Run the Consistency Check**
Processes the `train.csv` file and generates the final results in `results_json.csv`.

```powershell
python main4.py

```

## 📂 Project Layout

* `main4.py`: Top-level script for the primary logic pipeline (Fact Extraction -> Retrieval -> Judgment).
* `build_and_query.py`: Script to ingest books and build FAISS indices.
* `eval.py`: Evaluation script for testing logic performance.
* `data/`: Source text files used for building vector stores.
* `src2/`: Core library modules (`data_loader.py`, `embedding.py`, `vectorstore.py`).
* `faiss_store_*/`: Generated FAISS indexes (Binary files, excluded from Git).

## 🧠 Logic Approach

The system distinguishes **Signal** (Contradictions) from **Noise** (Different Phrasing) using a three-tiered check:

1. **Location State:** Can the character physically be here? (e.g., *Dungeon vs Paris*)
2. **Vital State:** Is the character alive? (e.g., *Died in Ch. 5 vs Alive in Ch. 10*)
3. **Allegiance State:** Does their loyalty match? (e.g., *Royalist vs Bonapartist*)

## 📄 License

MIT License

```



