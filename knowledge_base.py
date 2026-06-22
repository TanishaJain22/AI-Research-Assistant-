from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from duckdb_vectordb import DuckDbVectorDb

def get_knowledge_base():
    # Initialize the local embedder
    embedder = SentenceTransformerEmbedder(
        id="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Return Knowledge base initialized with our custom DuckDbVectorDb
    return Knowledge(
        vector_db=DuckDbVectorDb(
            table_name="research_docs",
            db_path="tmp/db/duckdb.db",
            embedder=embedder,
        ),
    )
