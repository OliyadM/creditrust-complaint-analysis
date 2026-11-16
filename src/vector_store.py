"""Embed chunks with MiniLM and keep them in a FAISS index on disk.

Embeddings are L2-normalised and stored in an inner-product index, so scores
are cosine similarities (same ranking as IndexFlatL2 on normalised vectors,
but the numbers are easier to read and threshold).

  python -m src.vector_store            # build from data/processed sample
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.chunking import chunk_complaints

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.parquet"
INFO_FILE = "info.json"


def load_embedder(name: str = config.EMBED_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


def embed(texts, model, batch_size: int = 128, show_progress: bool = False) -> np.ndarray:
    vecs = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=show_progress, convert_to_numpy=True)
    return vecs.astype("float32")


def build_index(chunks: pd.DataFrame, out_dir: Path = config.VECTOR_DIR, model=None,
                extra_info: dict | None = None) -> Path:
    import faiss

    model = model or load_embedder()
    t0 = time.time()
    vecs = embed(chunks["text"], model, show_progress=len(chunks) > 1000)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / INDEX_FILE))
    chunks.reset_index(drop=True).to_parquet(out_dir / CHUNKS_FILE, index=False)
    info = {
        "embed_model": config.EMBED_MODEL,
        "chunks": int(len(chunks)),
        "complaints": int(chunks["complaint_id"].nunique()),
        "dim": int(vecs.shape[1]),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embed_seconds": round(time.time() - t0, 1),
        "per_product_chunks": chunks["product"].value_counts().to_dict(),
        **(extra_info or {}),
    }
    (out_dir / INFO_FILE).write_text(json.dumps(info, indent=2))
    return out_dir


def load_index(path: Path = config.VECTOR_DIR):
    import faiss

    path = Path(path)
    index = faiss.read_index(str(path / INDEX_FILE))
    chunks = pd.read_parquet(path / CHUNKS_FILE)
    if index.ntotal != len(chunks):
        raise ValueError(f"index has {index.ntotal} vectors but {len(chunks)} chunk rows")
    return index, chunks


def index_exists(path: Path = config.VECTOR_DIR) -> bool:
    path = Path(path)
    return (path / INDEX_FILE).exists() and (path / CHUNKS_FILE).exists()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default=config.SAMPLE_PATH)
    ap.add_argument("--out", type=Path, default=config.VECTOR_DIR)
    args = ap.parse_args(argv)
    df = pd.read_csv(args.sample, dtype=str)
    chunks = chunk_complaints(df)
    print(f"{len(df):,} complaints -> {len(chunks):,} chunks")
    build_index(chunks, args.out, extra_info={"source": str(args.sample.name)})
    print((args.out / INFO_FILE).read_text())


if __name__ == "__main__":
    main()
