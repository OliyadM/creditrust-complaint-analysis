"""Make sure a vector store is available before the app starts.

Order of preference:
1. an index already on disk in vector_store/
2. a prebuilt index in a Hugging Face dataset repo (env CREDITRUST_INDEX_REPO),
   e.g. the 20k-complaint index built by `python -m src.vector_store`
3. a small demo index built on the spot from one CFPB archive file
   (two months of complaints, a few hundred per product). Slower to start,
   but needs nothing except internet access.
"""
from __future__ import annotations

import os
from pathlib import Path

from src import config
from src.vector_store import CHUNKS_FILE, INDEX_FILE, INFO_FILE, index_exists

DEMO_FILE = os.environ.get("CREDITRUST_DEMO_FILE", "CCDB_Export_12_July_2025_through_August_2025.zip")
DEMO_PER_PRODUCT = int(os.environ.get("CREDITRUST_DEMO_PER_PRODUCT", "500"))


def fetch_from_hub(repo_id: str, out_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(out_dir),
                      allow_patterns=[INDEX_FILE, CHUNKS_FILE, INFO_FILE],
                      token=os.environ.get("HF_TOKEN"))
    return out_dir


def build_demo_index(out_dir: Path, file_name: str = DEMO_FILE,
                     per_product: int = DEMO_PER_PRODUCT) -> Path:
    import pandas as pd
    import requests

    from src.chunking import chunk_complaints
    from src.ingest import FILE_BASE, HEADERS, filter_zip, stratified_sample
    from src.vector_store import build_index

    raw = config.RAW_DIR / file_name
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {file_name} for the demo index")
        with requests.get(FILE_BASE + file_name, headers=HEADERS, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(raw, "wb") as fh:
                for block in r.iter_content(1 << 20):
                    fh.write(block)
    df = pd.concat([chunk for _, _, chunk in filter_zip(raw)], ignore_index=True)
    sample = stratified_sample(df, per_product)
    chunks = chunk_complaints(sample)
    print(f"demo index: {len(sample)} complaints, {len(chunks)} chunks")
    return build_index(chunks, out_dir, extra_info={"source": f"demo sample from {file_name}"})


def ensure_index(out_dir: Path = config.VECTOR_DIR) -> Path:
    if index_exists(out_dir):
        return out_dir
    repo = os.environ.get("CREDITRUST_INDEX_REPO")
    if repo:
        print(f"fetching prebuilt index from {repo}")
        fetch_from_hub(repo, out_dir)
        if index_exists(out_dir):
            return out_dir
        print("hub download did not contain an index, falling back to the demo build")
    return build_demo_index(out_dir)
