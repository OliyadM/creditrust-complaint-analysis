import faiss
import numpy as np

from src.ingest import list_archive_zips, stratified_sample
from src.rag import (
    NO_EVIDENCE,
    Hit,
    RAGPipeline,
    Retriever,
    build_check_prompt,
    build_map_prompt,
    build_reduce_prompt,
    is_no_evidence,
)


def _retriever(chunks, embedder):
    vecs = embedder.encode(chunks["text"].tolist())
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(np.asarray(vecs, dtype="float32"))
    return Retriever(index, chunks, embedder)


def test_retrieval_finds_matching_product(tiny_chunks, embedder):
    hits = _retriever(tiny_chunks, embedder).search("late fee on my credit card", k=2)
    assert hits[0].product == "Credit card"
    # one chunk per complaint
    assert len({h.complaint_id for h in hits}) == len(hits)


def test_product_filter(tiny_chunks, embedder):
    hits = _retriever(tiny_chunks, embedder).search("late fee on my credit card", k=3,
                                                    product="Savings account")
    assert [h.product for h in hits] == ["Savings account"]


def test_check_prompt_asks_yes_no():
    hit = Hit("1", "Credit card", "Fees", "Bank", "2024-01-01", "late fee", 0.8)
    assert build_check_prompt("Why?", hit).endswith("Answer yes or no.")


def test_map_prompt_truncates_and_keeps_question():
    hit = Hit("123", "Credit card", "Fees", "Bank", "2024-01-01", "late fee " * 200, 0.8)
    prompt = build_map_prompt("  Why the fees? ", hit, max_chars=100)
    assert "Question: Why the fees?\n" in prompt
    assert "late fee " * 20 not in prompt
    assert prompt.count("late fee") <= 12
    assert prompt.rstrip().endswith("what problem does this customer describe?")


def test_reduce_prompt_lists_findings():
    prompt = build_reduce_prompt("Why the fees?", ["Late fee charged.", " ", "Interest added."])
    assert "- Late fee charged.\n- Interest added." in prompt
    assert "Question: Why the fees?" in prompt


def test_no_evidence_detection():
    assert is_no_evidence(NO_EVIDENCE)
    assert is_no_evidence("")
    assert not is_no_evidence("Customers complain about late fees.")


class EchoGenerator:
    """Relevance check is high only for excerpts that mention a fee."""

    def __init__(self):
        self.calls = 0

    def yes_probability(self, prompt):
        self.calls += 1
        return 0.9 if "fee" in prompt.split("Question:")[0] else 0.05

    def __call__(self, prompt, max_new_tokens=None):
        self.calls += 1
        return "Customers report late fees."


def test_pipeline_declines_on_low_score(tiny_chunks, embedder):
    gen = EchoGenerator()
    pipe = RAGPipeline(_retriever(tiny_chunks, embedder), gen, min_score=0.3)
    res = pipe.answer("chocolate cake recipe")
    assert res.refused and res.answer == NO_EVIDENCE and gen.calls == 0


def test_pipeline_uses_only_relevant_complaints(tiny_chunks, embedder):
    gen = EchoGenerator()
    pipe = RAGPipeline(_retriever(tiny_chunks, embedder), gen, min_score=0.1, min_relevance=0.2)
    res = pipe.answer("late fee on my credit card", k=3)
    assert not res.refused
    assert res.source_ids == ["1"]  # only the fee complaint passed the check
    assert res.relevant.count(True) == 1
    # k checks + one finding + one summary
    assert gen.calls == len(res.hits) + 2


def test_pipeline_declines_when_check_rejects(tiny_chunks, embedder):
    gen = EchoGenerator()
    pipe = RAGPipeline(_retriever(tiny_chunks, embedder), gen, min_score=0.1, min_relevance=0.5)
    res = pipe.answer("late fee on my credit card", k=3)
    assert res.refused and res.answer == NO_EVIDENCE
    assert res.relevance[0] == 0.9


def test_list_archive_zips():
    html = ('<a href="https://files.consumerfinance.gov/f/documents/CCDB_Export_2_May_2018.zip">x</a>'
            '<a href="https://files.consumerfinance.gov/f/documents/CCDB_Export_1_Dec_2011.zip">y</a>')
    assert len(list_archive_zips(html)) == 2


def test_stratified_sample_is_seeded_and_capped():
    import pandas as pd

    df = pd.DataFrame({"complaint_id": [str(i) for i in range(30)],
                       "product": ["A"] * 25 + ["B"] * 5,
                       "narrative": ["text XXXX here"] * 30})
    a = stratified_sample(df, 10)
    b = stratified_sample(df, 10)
    assert a["complaint_id"].tolist() == b["complaint_id"].tolist()
    assert a["product"].value_counts().to_dict() == {"A": 10, "B": 5}
    assert a["clean_narrative"].iloc[0] == "text here"
