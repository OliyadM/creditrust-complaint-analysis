import hashlib
import re

import numpy as np
import pandas as pd
import pytest


class HashEmbedder:
    """Tiny bag-of-words embedder so retrieval can be tested without a model download."""

    dim = 256

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **_):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim
                out[i, h] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def tiny_chunks():
    return pd.DataFrame([
        {"complaint_id": "1", "product": "Credit card", "issue": "Fees", "company": "Bank A",
         "date_received": "2024-01-01", "chunk_no": 0,
         "text": "I was charged a late fee on my credit card even though I paid on time."},
        {"complaint_id": "1", "product": "Credit card", "issue": "Fees", "company": "Bank A",
         "date_received": "2024-01-01", "chunk_no": 1,
         "text": "The late fee and interest charge were never reversed by the card issuer."},
        {"complaint_id": "2", "product": "Money transfers", "issue": "Fraud", "company": "Pay Co",
         "date_received": "2024-02-01", "chunk_no": 0,
         "text": "I sent money through the app to a scammer and the transfer was not refunded."},
        {"complaint_id": "3", "product": "Savings account", "issue": "Withdrawal", "company": "Bank B",
         "date_received": "2024-03-01", "chunk_no": 0,
         "text": "The bank froze my savings account and I could not withdraw my money."},
        {"complaint_id": "4", "product": "Buy Now, Pay Later (BNPL)", "issue": "Refund",
         "company": "Klarna AB", "date_received": "2024-04-01", "chunk_no": 0,
         "text": "I returned the item but the pay later plan still charged my installment."},
    ])
