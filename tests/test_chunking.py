import pandas as pd

from src.chunking import chunk_complaints


def _df(text):
    return pd.DataFrame([{"complaint_id": "9", "product": "Credit card", "issue": "Fees",
                          "company": "Bank", "date_received": "2024-01-01", "clean_narrative": text}])


def test_short_text_is_one_chunk():
    out = chunk_complaints(_df("Short complaint."))
    assert len(out) == 1
    assert out.loc[0, "text"] == "Short complaint."
    assert out.loc[0, "complaint_id"] == "9"


def test_long_text_is_split_with_metadata_and_size_limit():
    text = " ".join(f"Sentence number {i} about a disputed charge." for i in range(80))
    out = chunk_complaints(_df(text), size=200, overlap=20)
    assert len(out) > 5
    assert out["text"].str.len().max() <= 200
    assert set(out["product"]) == {"Credit card"}
    assert out["chunk_no"].tolist() == list(range(len(out)))


def test_empty_narratives_are_skipped():
    df = pd.concat([_df(""), _df(None), _df("ok")])
    assert len(chunk_complaints(df)) == 1
