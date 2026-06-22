import os
from dotenv import load_dotenv
from knowledge_base import get_knowledge_base

load_dotenv()

DATA_DIR = "data"
SUPPORTED_EXTENSIONS = {".txt": "TXT", ".md": "MD"}

def run_ingestion():
    if not os.path.isdir(DATA_DIR):
        print(f"No '{DATA_DIR}/' directory found. Add .txt or .md files there first.")
        return

    files = sorted(os.listdir(DATA_DIR))
    print("Initializing Knowledge Base...")
    kb = get_knowledge_base()

    ingested = 0
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        path = os.path.join(DATA_DIR, filename)
        print(f"[{SUPPORTED_EXTENSIONS[ext]}] Indexing {filename}...")
        kb.add_content(path=path, upsert=True)
        ingested += 1

    if ingested == 0:
        print(f"No supported files (.txt, .md) found in '{DATA_DIR}/'.")
    else:
        print(f"\n[SUCCESS] {ingested} document(s) ingested into the local DuckDB vector DB!")
        print("          Database path: tmp/db/duckdb.db")

if __name__ == "__main__":
    run_ingestion()
