import os
import time
import pandas as pd
import ftfy
import json

from groq import Groq, RateLimitError, APIError
from dotenv import load_dotenv


# Import your custom modules
from src2.vectorstore import FaissVectorStore
from src2.embedding import EmbeddingPipeline

# ==========================================
# 1. CONFIGURATION
# ==========================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY_2") 
client = Groq(api_key=GROQ_API_KEY)

# UPDATED PRIORITY LIST based on your Rate Limits
MODEL_LIST = [
    "qwen/qwen3-32b",             # PRIORITY 1: 500k Tokens/Day (Smart & High Limit)
     "llama-3.1-8b-instant",       # PRIORITY 2: 500k Tokens/Day (Fast Backup)
    "openai/gpt-oss-120b",        # PRIORITY 3: 200k Tokens/Day (Very Smart Backup)
    "llama-3.3-70b-versatile"     # PRIORITY 4: 100k Tokens/Day (Last Resort)
]

RETRIEVAL_STRATEGY = "STORY" 
INPUT_CSV = "train.csv"
OUTPUT_CSV = "resultsSTORY.csv"

STORE_DIR_MC = "./faiss_store_monte_cristo"
STORE_DIR_CASTAWAYS = "./faiss_store_castaways"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def clean_text(text):
    if pd.isna(text): return ""
    return ftfy.fix_text(str(text)).strip()

def safe_api_call_with_fallback(messages, max_tokens=1000, temperature=0.0, json_mode=False):
    """
    Tries models in order. If one fails, it swaps to the next in the list.
    """
    for model in MODEL_LIST:
        for attempt in range(3): 
            try:
                # 4s wait is safe for 60 RPM models
                time.sleep(4) 
                
                params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                if json_mode:
                    params["response_format"] = {"type": "json_object"}

                response = client.chat.completions.create(**params)
                return response.choices[0].message.content

            except RateLimitError:
                print(f"   [!] Rate Limit on {model}. Switching models...")
                break # Break loop to try NEXT model
            except APIError as e:
                print(f"   [!] API Error on {model}: {e}")
                time.sleep(5)
            except Exception as e:
                # If model not found, print and switch
                print(f"   [!] Error on {model}: {e}")
                break
                
    print("   [CRITICAL] All models failed.")
    return None

def load_stores():
    print("--- Loading Vector Stores ---")
    stores = {}
    if os.path.exists(STORE_DIR_MC):
        mc = FaissVectorStore(STORE_DIR_MC)
        mc.load()
        stores["The Count of Monte Cristo"] = mc
    
    if os.path.exists(STORE_DIR_CASTAWAYS):
        cast = FaissVectorStore(STORE_DIR_CASTAWAYS)
        cast.load()
        stores["In Search of the Castaways"] = cast
    return stores

# ==========================================
# 3. LOGIC (Optimized)
# ==========================================
def extract_facts(backstory,char_name,caption="" ):
    """
    Uses LLM to break backstory into facts (JSON Mode).
    """
    prompt = f"""
    You are a Fact Extractor.
    Break the text into a JSON list of atomic facts.
    
    TEXT: "This is about a character named {char_name}, the caption if present is {caption}, and the backstory is {backstory}"
    
    OUTPUT FORMAT:
    {{
        "facts": ["Fact 1", "Fact 2"]
    }}
    """
    
    # Force JSON output for the facts too
    response = safe_api_call_with_fallback([{"role": "user", "content": prompt}], json_mode=True)
    
    if not response:
        return [backstory]
        
    try:
        data = json.loads(response)
        return data.get("facts", [backstory])
    except:
        return [backstory]

embedder = None

def get_evidence_for_row(row, store, facts):
    char_name = clean_text(row.get('char', ''))
    caption = clean_text(row.get('caption', ''))
    
    collected_chunks = {} 
    
    # 1. Search
    for fact in facts:
        q = f"{char_name} ({caption}): {fact}"
        vec = embedder.model.encode([q])
        results = store.search(vec, top_k=3)
        
        for r in results:
            cid = r['chunk_id']
            if cid not in collected_chunks:
                collected_chunks[cid] = r
    
    # 2. CAP EVIDENCE (Top 10 chunks to save tokens)
    all_chunks = list(collected_chunks.values())
    all_chunks.sort(key=lambda x: x['distance']) # Sort by relevance
    top_chunks = all_chunks[:10]                 # Keep top 10
    top_chunks.sort(key=lambda x: x['chunk_id']) # Sort by Timeline
    
    return top_chunks

def verify_consistency(row, store):
    """
    Returns Dictionary: {'prediction': 0/1, 'rationale': '...'}
    """
    backstory = clean_text(row.get('content', ''))
    char_name = clean_text(row.get('char', 'Unknown Character'))
    caption = clean_text(row.get('caption', 'No Caption'))
    
    # 1. Extract Facts (Locally)
    if RETRIEVAL_STRATEGY == "FACTS":
        facts = extract_facts(backstory,char_name,caption)
    else:
        facts = [backstory]

    print(f"   -> Checking {len(facts)} facts (Regex)...")
        
    # 2. Retrieve Evidence
    evidence_items = get_evidence_for_row(row, store, facts)
    
    if not evidence_items:
        return {"prediction": 1, "rationale": "No evidence found (Benefit of Doubt)"}
    
    evidence_text = ""
    for item in evidence_items:
        evidence_text += f"[Chunk {item['chunk_id']}]: \"{item['text']}\"\n\n"
        
    # 3. THE SMART JUDGE PROMPT
    prompt = f"""
    You are a Narrative Logic Judge.
    
    TASK: Determine if the [BACKSTORY] contradicts the [BOOK EXCERPTS].
    
    [CHARACTER]: {char_name}
    [CAPTION]: {caption}
    
    [BACKSTORY CLAIM]
    {backstory}
    
    [BOOK EXCERPTS (Timeline Ordered)]
    {evidence_text}
    
    [CRITICAL ANALYSIS RULES]
    1. **CHECK MUTUALLY EXCLUSIVE STATES**:
       - **Location**: If the Book places the character in Prison, and Claim places them in Paris -> CONTRADICTION (0).
       - **Life/Death**: If Book says they died, and Claim says they did something later -> CONTRADICTION (0).
       - **Alliance**: If Book says they are a Royalist, and Claim says they are a Bonapartist -> CONTRADICTION (0).
       
    2. **SILENCE IS NOT A CONTRADICTION**:
       - If the Book merely *doesn't mention* the event, return 1.
       - Only return 0 if the Book actively establishes a reality where the Claim is IMPOSSIBLE.

    3. **TIMELINE LOGIC**:
       - Pay attention to the sequence of events implied by the [CAPTION] and text.
    4. **character check**:
       - Ensure the character mentioned in the backstory aligns with the character in the book excerpts. If not, ignore that evidence text.
    
    [CRITICAL OUTPUT REQUIREMENT]
    Return a JSON object:
    {{
        "prediction": 0 or 1,
        "rationale": "Identify the conflict: [Book State] vs [Claim State]. Quote the text."
    }}
    """
    
    # Use the FALLBACK function
    output = safe_api_call_with_fallback([{"role": "user", "content": prompt}], json_mode=True)
    
    result = {"prediction": 1, "rationale": "API Error"}
    if output:
        try:
            result = json.loads(output)
            # Ensure keys exist
            if "prediction" not in result: result["prediction"] = 1
        except json.JSONDecodeError:
            result = {"prediction": 1, "rationale": "JSON Parse Error"}
            
    return result

# ==========================================
# 4. MAIN
# ==========================================
def main():
    global embedder
    
    stores = load_stores()
    if not stores: return

    print("--- Loading Embedder ---")
    embedder = EmbeddingPipeline()
    
    df = pd.read_csv(INPUT_CSV)
    if 'content' not in df.columns and 'backstory' in df.columns:
        df['content'] = df['backstory']
        
    results = []
    print(f"--- Processing {len(df)} rows ---")
    
    for i, (idx, row) in enumerate(df.iterrows()):
        story_id = row.get('id', idx)
        book_name = row.get('book_name', '').strip()
        
        print(f"[{i+1}/{len(df)}] ID {story_id}...", end=" ", flush=True)
        
        store = None
        for key in stores:
            if key in book_name:
                store = stores[key]
                break
        
        if not store:
            print("Skip")
            continue
            
        output_dict = verify_consistency(row, store)
        
        pred = output_dict.get("prediction", 1)
        rationale = output_dict.get("rationale", "")
        
        print(f"-> {pred}")
        
        results.append({
            "Story ID": story_id,
            "Prediction": pred,
            "Rationale": rationale
        })
        
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()