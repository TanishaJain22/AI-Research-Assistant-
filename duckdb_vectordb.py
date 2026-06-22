import os
import json
import duckdb
from typing import List, Dict, Any, Optional
from agno.vectordb.base import VectorDb
from agno.knowledge.document import Document
from agno.knowledge.embedder.base import Embedder

class DuckDbVectorDb(VectorDb):
    def __init__(
        self,
        table_name: str = "research_docs",
        db_path: str = "tmp/db/duckdb.db",
        embedder: Optional[Embedder] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.table_name = table_name
        self.db_path = db_path
        self.embedder = embedder
        
        # Ensure parent directory exists and create table
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.create()

    def _conn(self):
        return duckdb.connect(self.db_path)

    def _value_exists(self, column: str, value: Any) -> bool:
        with self._conn() as conn:
            res = conn.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE {column} = ?", (value,)).fetchone()
            return res[0] > 0 if res else False

    def create(self) -> None:
        with self._conn() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id VARCHAR, name VARCHAR, meta_data VARCHAR, 
                    content VARCHAR, embedding FLOAT[], content_hash VARCHAR
                )
            """)

    def name_exists(self, name: str) -> bool: return self._value_exists("name", name)
    def id_exists(self, id: str) -> bool: return self._value_exists("id", id)
    def content_hash_exists(self, content_hash: str) -> bool: return self._value_exists("content_hash", content_hash)
    
    def insert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        with self._conn() as conn:
            for doc in documents:
                if not doc.embedding and self.embedder:
                    doc.embedding = self.embedder.get_embedding(doc.content)
                conn.execute(
                    f"INSERT INTO {self.table_name} (id, name, meta_data, content, embedding, content_hash) VALUES (?, ?, ?, ?, ?, ?)",
                    (doc.id, doc.name, json.dumps(doc.meta_data or {}), doc.content, doc.embedding, content_hash)
                )

    def upsert_available(self) -> bool: return True

    def upsert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        with self._conn() as conn:
            conn.execute(f"DELETE FROM {self.table_name} WHERE content_hash = ?", (content_hash,))
        self.insert(content_hash, documents, filters)

    def search(self, query: str, limit: int = 5, filters: Optional[Any] = None) -> List[Document]:
        if not self.embedder: raise ValueError("Embedder must be provided to run search.")
        
        query_embedding = self.embedder.get_embedding(query)
        with self._conn() as conn:
            res = conn.execute(f"""
                SELECT id, name, meta_data, content, list_cosine_similarity(embedding, ?::FLOAT[]) as similarity
                FROM {self.table_name} ORDER BY similarity DESC LIMIT ?
            """, (query_embedding, limit)).fetchall()
        
        docs = []
        for row in res:
            meta = json.loads(row[2]) if row[2] else {}
            meta["score"] = row[4]
            docs.append(Document(id=row[0], name=row[1], meta_data=meta, content=row[3]))
        return docs

    def exists(self) -> bool:
        with self._conn() as conn:
            res = conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", (self.table_name,)).fetchone()
            return res[0] > 0 if res else False

    def get_supported_search_types(self) -> List[str]: return ["vector"]

    # --- Abstract Method Implementations (Required by ABC) ---
    # Class-level aliases map async methods straight to sync methods.
    async_create = create
    async_name_exists = name_exists
    async_id_exists = id_exists
    async_content_hash_exists = content_hash_exists
    async_insert = insert
    async_upsert = upsert
    async_search = search
    async_exists = exists

    # Unused required stubs condensed
    def drop(self): pass
    def async_drop(self): pass
    def delete(self): return False
    def delete_by_id(self, id: str): return False
    def delete_by_name(self, name: str): return False
    def delete_by_metadata(self, metadata): return False
    def delete_by_content_id(self, content_id: str): return False
