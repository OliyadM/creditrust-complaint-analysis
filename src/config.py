"""Shared paths and defaults. Everything is relative to the repo root so the
scripts and the Gradio app behave the same no matter where they are launched."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("CREDITRUST_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
FILTERED_PATH = DATA_DIR / "filtered" / "filtered_complaints.csv.gz"
SAMPLE_PATH = DATA_DIR / "processed" / "sample_complaints.csv.gz"

VECTOR_DIR = Path(os.environ.get("CREDITRUST_VECTOR_DIR", ROOT / "vector_store"))
REPORTS_DIR = ROOT / "reports"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = os.environ.get("CREDITRUST_GEN_MODEL", "google/flan-t5-base")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
SEED = 42
