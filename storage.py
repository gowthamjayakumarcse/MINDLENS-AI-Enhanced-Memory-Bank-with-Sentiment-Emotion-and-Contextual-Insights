
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import uuid
import json
import numpy as np

from config import VECTOR_STORE, CHROMA_DIR, FAISS_INDEX_PATH, FAISS_META_JSONL, ENTRIES_JSONL

# Optional deps
try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None

try:
    import faiss                   # type: ignore
except Exception:
    faiss = None

@dataclass
class DiaryRecord:
    doc_id: str
    date: str
    text: str
    embedding: List[float]
    sentiment: str
    emotions: List[str]
    tags: List[str]
    # Optional image metadata
    image_path: Optional[str] = None  # relative path to stored image
    image_desc: Optional[str] = None  # user-provided description for retrieval
    # Optional video metadata
    video_path: Optional[str] = None  # relative path to stored video

class VectorStore:
    def __init__(self):
        self.backend = VECTOR_STORE.lower()
        if self.backend == "chroma":
            if chromadb is None:
                raise RuntimeError("chromadb package not installed.")
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self.col = self.client.get_or_create_collection(name="mindlens", metadata={"hnsw:space": "cosine"})
        elif self.backend == "faiss":
            if faiss is None:
                raise RuntimeError("faiss package not installed.")
            FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.index = None
            self.metadata: List[DiaryRecord] = []
            # load if exists
            if FAISS_INDEX_PATH.exists():
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            if FAISS_META_JSONL.exists():
                with open(FAISS_META_JSONL, "r", encoding="utf-8") as f:
                    self.metadata = [DiaryRecord(**json.loads(line)) for line in f]
        else:
            raise ValueError("Unsupported VECTOR_STORE: use 'chroma' or 'faiss'")

        ENTRIES_JSONL.parent.mkdir(parents=True, exist_ok=True)

    def reload(self):
        """Reload the vector store from disk (useful after external modifications)."""
        if self.backend == "chroma":
            # ChromaDB reloads automatically from disk
            pass
        elif self.backend == "faiss":
            # Reload FAISS index and metadata from disk
            self.index = None
            self.metadata = []
            if FAISS_INDEX_PATH.exists():
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            if FAISS_META_JSONL.exists():
                with open(FAISS_META_JSONL, "r", encoding="utf-8") as f:
                    self.metadata = [DiaryRecord(**json.loads(line)) for line in f]

    def upsert(self, records: List[DiaryRecord]):
        if self.backend == "chroma":
            ids = [r.doc_id for r in records]
            embs = [r.embedding for r in records]
            metas = [{
                "date": r.date,
                "sentiment": r.sentiment,
                "emotions": r.emotions,
                "tags": r.tags,
                "text": r.text,
                "image_path": r.image_path,
                "image_desc": r.image_desc,
                "video_path": r.video_path,
            } for r in records]
            self.col.upsert(ids=ids, embeddings=embs, metadatas=metas, documents=[r.text for r in records])
        else:
            # FAISS
            import numpy as np
            vecs = np.array([r.embedding for r in records]).astype("float32")
            if self.index is None:
                d = vecs.shape[1]
                self.index = faiss.IndexFlatIP(d)
            self.index.add(vecs)
            # append metadata
            with open(FAISS_META_JSONL, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            faiss.write_index(self.index, str(FAISS_INDEX_PATH))

        # append to entries jsonl for analytics
        with open(ENTRIES_JSONL, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def query(self, query_vec: List[float], top_k: int = 5,
              where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        where = where or {}
        if self.backend == "chroma":
            res = self.col.query(query_embeddings=[query_vec], n_results=top_k, where=where, include=["metadatas", "documents", "distances", "ids"])
            # normalize output
            out = []
            for i in range(len(res["ids"][0])):
                meta = res["metadatas"][0][i]
                out.append({
                    "doc_id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "distance": res["distances"][0][i],
                    **meta,
                })
            return out
        else:
            # FAISS: client-side filtering
            import numpy as np, json
            if self.index is None:
                return []
            q = np.array(query_vec, dtype="float32")[None, :]
            distances, idxs = self.index.search(q, top_k * 10)  # overfetch, then filter
            matches = []
            # read metadata list
            metas = []
            if Path(FAISS_META_JSONL).exists():
                with open(FAISS_META_JSONL, "r", encoding="utf-8") as f:
                    for line in f:
                        metas.append(json.loads(line))
            for dist, idx in zip(distances[0], idxs[0]):
                if idx == -1:
                    continue
                meta = metas[idx]
                # apply where filters (tags/emotions intersection checks)
                ok = True
                if "tags" in where:
                    cond = where["tags"]
                    if isinstance(cond, dict) and "$contains" in cond:
                        needed = cond["$contains"]
                    else:
                        needed = cond
                    needed = set(needed) if isinstance(needed, list) else {needed}
                    meta_tags = set(meta.get("tags", []))
                    if not needed.intersection(meta_tags):  # Check if any tag matches
                        ok = False
                if "emotions" in where and ok:
                    cond = where["emotions"]
                    if isinstance(cond, dict) and "$contains" in cond:
                        needed = cond["$contains"]
                    else:
                        needed = cond
                    needed = set(needed) if isinstance(needed, list) else {needed}
                    meta_emotions = set(meta.get("emotions", []))
                    if not needed.intersection(meta_emotions):  # Check if any emotion matches
                        ok = False
                if ok:
                    matches.append({**meta, "distance": float(dist)})
                if len(matches) >= top_k:
                    break
            return matches

def new_doc_id() -> str:
    return str(uuid.uuid4())
