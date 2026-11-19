"""Upload the built vector store to a Hugging Face dataset repo so the Space can
load it at startup (see src/bootstrap.py).

  HF_TOKEN=... python -m src.publish_index --repo <user>/creditrust-complaint-index
"""
from __future__ import annotations

import argparse
import os

from src import config
from src.vector_store import CHUNKS_FILE, INDEX_FILE, INFO_FILE, index_exists


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args(argv)
    if not index_exists(config.VECTOR_DIR):
        raise SystemExit("No index in vector_store/; run `python -m src.vector_store` first.")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(config.VECTOR_DIR), repo_id=args.repo, repo_type="dataset",
                      allow_patterns=[INDEX_FILE, CHUNKS_FILE, INFO_FILE],
                      commit_message="Upload complaint index")
    print(f"uploaded to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
