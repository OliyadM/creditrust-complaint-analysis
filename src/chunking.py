"""Split complaint narratives into overlapping chunks that keep their metadata."""
from __future__ import annotations

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

META_COLS = ["complaint_id", "product", "issue", "company", "date_received"]


def make_splitter(size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_complaints(df: pd.DataFrame, text_col: str = "clean_narrative",
                     size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> pd.DataFrame:
    """One row per chunk with the chunk text and the parent complaint's metadata."""
    splitter = make_splitter(size, overlap)
    rows = []
    for rec in df[META_COLS + [text_col]].itertuples(index=False):
        rec = rec._asdict()
        text = rec.pop(text_col)
        if not isinstance(text, str) or not text.strip():
            continue
        for i, piece in enumerate(splitter.split_text(text)):
            rows.append({**rec, "chunk_no": i, "text": piece})
    return pd.DataFrame(rows, columns=META_COLS + ["chunk_no", "text"])
