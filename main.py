from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rapidfuzz import fuzz
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

APP_DIR = Path(__file__).parent
CSV_PATH = APP_DIR / "health_1000_phrases_NT_kriol_STRICT.csv"

def load_phrase_bank(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Phrase bank CSV not found at: {path}")
    df = pd.read_csv(path)

    # Normalize expected columns
    cols = {c.lower().strip(): c for c in df.columns}
    # tolerate variations
    id_col = cols.get("id")
    ctx_col = cols.get("context_name") or cols.get("context") or cols.get("subcontext") or cols.get("contextname")
    type_col = cols.get("type") or cols.get("intent")
    en_col = cols.get("english")
    kr_col = cols.get("kriol") or cols.get("kriol")

    if en_col is None or kr_col is None:
        raise ValueError(f"CSV must contain 'english' and 'kriol' columns. Found: {list(df.columns)}")

    if id_col is None:
        df["id"] = [f"P{i+1:05d}" for i in range(len(df))]
        id_col = "id"
    if ctx_col is None:
        df["context_name"] = "General"
        ctx_col = "context_name"
    if type_col is None:
        df["type"] = "unknown"
        type_col = "type"

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "phrase_id": str(r[id_col]),
            "context_name": str(r[ctx_col]),
            "type": str(r[type_col]),
            "english": str(r[en_col]),
            "kriol": str(r[kr_col]),
        })
    return rows

PHRASES: List[Dict[str, Any]] = load_phrase_bank(CSV_PATH)

def best_matches(query: str, field: str, top_k: int = 5) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    scored = []
    for p in PHRASES:
        score = fuzz.token_set_ratio(q.lower(), (p.get(field) or "").lower())
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, p in scored[:max(1, top_k)]:
        out.append({
            "phrase_id": p["phrase_id"],
            "english": p["english"],
            "kriol": p["kriol"],
            "context_name": p["context_name"],
            "type": p["type"],
            "score": float(score),
        })
    return out

class KriolToEnglishReq(BaseModel):
    kriol_text: str
    top_k: int = 5

class EnglishToKriolReq(BaseModel):
    english_text: str
    top_k: int = 5

app = FastAPI(title="Ngurra Clinic Translator API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "phrases_loaded": len(PHRASES)}

@app.post("/translate/kriol-to-english")
def translate_kriol_to_english(req: KriolToEnglishReq):
    candidates = best_matches(req.kriol_text, field="kriol", top_k=req.top_k)
    return {"best": candidates[0] if candidates else None, "candidates": candidates}

@app.post("/translate/english-to-kriol")
def translate_english_to_kriol(req: EnglishToKriolReq):
    candidates = best_matches(req.english_text, field="english", top_k=req.top_k)
    return {"best": candidates[0] if candidates else None, "candidates": candidates}

@app.get("/suggestions")
def suggestions(
    context_name: str = Query(...),
    top_k: int = Query(20, ge=1, le=200)
):
    rows = [p for p in PHRASES if p["context_name"] == context_name]
    rows = rows[:top_k]
    return {
        "context_name": context_name,
        "suggestions": [
            {
                "phrase_id": p["phrase_id"],
                "english": p["english"],
                "kriol": p["kriol"],
                "type": p["type"],
            } for p in rows
        ]
    }
