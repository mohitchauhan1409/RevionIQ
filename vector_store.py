import json
import threading
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = Path(__file__).parent / "data" / "repair_cases.json"
CHROMA_PATH = Path(__file__).parent / "chroma_db"

# Heavy ML libs (torch, transformers, chromadb) are imported lazily so the
# process binds its port before any slow initialization runs.
_model = None  # SentenceTransformer, loaded on first use


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _embed(text: str) -> list[float]:
    return _get_model().encode(text, normalize_embeddings=True).tolist()


def _case_to_text(case: dict) -> str:
    v = case["vehicle"]
    return (
        f"{v['year']} {v['make']} {v['model']} {v.get('engine', '')} "
        f"fault:{' '.join(case['fault_codes'])} "
        f"complaint:{case['c1_complaint']} "
        f"cause:{case['c2_cause']} "
        f"correction:{case['c3_correction']} "
        f"notes:{case.get('technician_notes', '')}"
    )


class VectorStore:
    def __init__(self):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_or_create_collection(
            name="repair_cases",
            metadata={"hnsw:space": "cosine"},
        )
        self._loaded_count = 0
        self._ensure_loaded()

    def _ensure_loaded(self):
        existing = self.collection.count()
        if existing > 0:
            self._loaded_count = existing
            return

        with open(DATA_PATH) as f:
            data = json.load(f)

        cases = data["repair_cases"]
        ids, embeddings, documents, metadatas = [], [], [], []

        for case in cases:
            text = _case_to_text(case)
            embedding = _embed(text)
            ids.append(case["id"])
            embeddings.append(embedding)
            documents.append(text)
            metadatas.append({
                "make": case["vehicle"]["make"],
                "model": case["vehicle"]["model"],
                "year": case["vehicle"]["year"],
                "fault_codes": ",".join(case["fault_codes"]),
                "outcome": case.get("outcome", ""),
                "warranty_approved": str(case.get("warranty_approved", True)),
                "labor_hours": case.get("labor_hours", 0),
                "c1": case["c1_complaint"][:500],
                "c2": case["c2_cause"][:500],
                "c3": case["c3_correction"][:500],
                "technician_notes": case.get("technician_notes", ""),
                "common_mistakes": "; ".join(case.get("common_mistakes", [])),
            })

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        self._loaded_count = len(cases)

    def search_similar_cases(
        self,
        fault_codes: list[str],
        vehicle_make: str,
        vehicle_model: str,
        symptom: str,
        n: int = 5,
    ) -> list[dict]:
        query = f"{vehicle_make} {vehicle_model} fault:{' '.join(fault_codes)} {symptom}"
        query_embedding = _embed(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, self.collection.count()),
            include=["metadatas", "distances", "documents"],
        )

        cases = []
        for i, meta in enumerate(results["metadatas"][0]):
            cases.append({
                "make": meta["make"],
                "model": meta["model"],
                "year": meta["year"],
                "fault_codes": meta["fault_codes"].split(","),
                "c1": meta["c1"],
                "c2": meta["c2"],
                "c3": meta["c3"],
                "technician_notes": meta["technician_notes"],
                "common_mistakes": meta["common_mistakes"].split("; ") if meta["common_mistakes"] else [],
                "outcome": meta["outcome"],
                "warranty_approved": meta["warranty_approved"] == "True",
                "similarity_score": round(1 - results["distances"][0][i], 3),
            })
        return cases

    @property
    def case_count(self) -> int:
        return self._loaded_count


_store: Optional[VectorStore] = None
_store_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = VectorStore()
    return _store
