"""Retrieval-augmented answering over the complaint index.

Retrieval: MiniLM query embedding -> FAISS top-k chunks (one chunk per
complaint so the context is not five slices of the same story).
Generation: a local seq2seq model (flan-t5-base by default) that only sees the
retrieved excerpts, in three short steps:
  1. check  - for each retrieved complaint, the model's probability of "yes"
              to "is this complaint related to the topic of the question?".
              Mean below MIN_RELEVANCE -> decline; complaints below
              HIT_RELEVANCE are shown but not used.
  2. map    - for each relevant complaint, one sentence on the problem it
              describes (the "findings", each tied to its complaint ID)
  3. reduce - a two-sentence answer written only from those findings
Small models handle one complaint at a time much better than five pasted
together: with a single long prompt flan-t5 mostly copied one excerpt.

Questions whose best retrieval score is below MIN_SCORE are declined without
calling the generator. A similarity threshold alone could not separate
on-topic from off-topic questions (they overlap around 0.6), which is why
the relevance check exists. The thresholds were picked on the dev questions in
eval/dev_questions.json, not on the evaluation set.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config

NO_EVIDENCE = "Not enough evidence in the retrieved complaints to answer this."
# Cosine similarity below which a question is declined outright.
MIN_SCORE = 0.40
# Relevance check: mean P(yes) over the retrieved complaints needed to answer,
# and per-complaint P(yes) needed for a complaint to be used. On the dev set the
# on-topic means were 0.34-0.87 and off-topic 0.00-0.26 (plus one how-to
# question about online banking at 0.53 that still gets answered).
MIN_RELEVANCE = 0.30
HIT_RELEVANCE = 0.30
MAX_EXCERPT_CHARS = 600

CHECK_TEMPLATE = (
    "Customer complaint: {text}\n\n"
    "Question: {question}\n"
    "Is this complaint related to the topic of the question? Answer yes or no."
)

MAP_TEMPLATE = (
    "Customer complaint: {text}\n\n"
    "Question: {question}\n"
    "In one sentence, what problem does this customer describe?"
)
REDUCE_TEMPLATE = (
    "Customer problems:\n{findings}\n\n"
    "Question: {question}\n"
    "Using only the problems listed above, write a two-sentence answer that "
    "summarizes what customers complain about."
)


@dataclass
class Hit:
    complaint_id: str
    product: str
    issue: str
    company: str
    date_received: str
    text: str
    score: float


@dataclass
class RAGResult:
    question: str
    answer: str
    hits: list[Hit] = field(default_factory=list)
    relevance: list[float] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    refused: bool = False

    @property
    def relevant(self) -> list[bool]:
        return [p >= HIT_RELEVANCE for p in self.relevance]

    @property
    def source_ids(self) -> list[str]:
        """IDs of the complaints the answer was written from."""
        if self.findings:
            return [h.complaint_id for h, f in zip(self.hits, self.findings, strict=True) if f]
        return [h.complaint_id for h in self.hits]


def truncate(text: str, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def build_check_prompt(question: str, hit: Hit, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    return CHECK_TEMPLATE.format(text=truncate(hit.text, max_chars), question=question.strip())


def build_map_prompt(question: str, hit: Hit, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    return MAP_TEMPLATE.format(text=truncate(hit.text, max_chars), question=question.strip())


def build_reduce_prompt(question: str, findings: list[str]) -> str:
    bullets = "\n".join(f"- {f.strip()}" for f in findings if f.strip())
    return REDUCE_TEMPLATE.format(findings=bullets, question=question.strip())


def is_no_evidence(text: str) -> bool:
    t = text.lower()
    return "not enough evidence" in t or t.strip() in {"", "none", "unanswerable", "no"}


class Retriever:
    def __init__(self, index, chunks: pd.DataFrame, embedder):
        self.index = index
        self.chunks = chunks.reset_index(drop=True)
        self.embedder = embedder

    def encode(self, text: str) -> np.ndarray:
        vec = self.embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return vec.astype("float32")

    def search(self, query: str, k: int = config.TOP_K, product: str | None = None) -> list[Hit]:
        q = self.encode(query)
        n = self.index.ntotal
        depth = min(n, max(k * 10, 50) if product is None else max(k * 200, 1000))
        while True:
            scores, ids = self.index.search(q, depth)
            hits, seen = [], set()
            for score, idx in zip(scores[0], ids[0], strict=True):
                if idx < 0:
                    continue
                row = self.chunks.iloc[int(idx)]
                if product and row["product"] != product:
                    continue
                if row["complaint_id"] in seen:
                    continue
                seen.add(row["complaint_id"])
                hits.append(Hit(
                    complaint_id=str(row["complaint_id"]), product=row["product"],
                    issue=str(row["issue"]), company=str(row["company"]),
                    date_received=str(row["date_received"]), text=row["text"],
                    score=float(score),
                ))
                if len(hits) == k:
                    return hits
            if depth >= n:
                return hits
            depth = min(n, depth * 4)


class Generator:
    def __init__(self, model_name: str = config.GEN_MODEL, max_new_tokens: int = 128):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str, max_new_tokens: int | None = None) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                num_beams=2,
                no_repeat_ngram_size=3,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def yes_probability(self, prompt: str) -> float:
        """P("yes") / (P("yes") + P("no")) for the first generated token."""
        import torch

        yes = self.tokenizer("yes").input_ids[0]
        no = self.tokenizer("no").input_ids[0]
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        start = torch.tensor([[self.model.config.decoder_start_token_id]])
        with torch.no_grad():
            logits = self.model(**inputs, decoder_input_ids=start).logits[0, -1]
        return float(torch.softmax(logits[[yes, no]], dim=0)[0])


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator, min_score: float = MIN_SCORE,
                 min_relevance: float = MIN_RELEVANCE):
        self.retriever = retriever
        self.generator = generator
        self.min_score = min_score
        self.min_relevance = min_relevance

    @classmethod
    def from_disk(cls, vector_dir=config.VECTOR_DIR, gen_model: str = config.GEN_MODEL):
        from src.vector_store import load_embedder, load_index

        index, chunks = load_index(vector_dir)
        return cls(Retriever(index, chunks, load_embedder()), Generator(gen_model))


    def answer(self, question: str, k: int = config.TOP_K, product: str | None = None) -> RAGResult:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k=k, product=product)
        t1 = time.perf_counter()
        if not hits or hits[0].score < self.min_score:
            return RAGResult(question, NO_EVIDENCE, hits, retrieval_s=t1 - t0, refused=True)
        relevance = [self.generator.yes_probability(build_check_prompt(question, h)) for h in hits]
        if sum(relevance) / len(relevance) < self.min_relevance:
            return RAGResult(question, NO_EVIDENCE, hits, relevance, retrieval_s=t1 - t0,
                             generation_s=time.perf_counter() - t1, refused=True)
        use = [p >= HIT_RELEVANCE for p in relevance]
        if not any(use):
            use[relevance.index(max(relevance))] = True
        findings = [self.generator(build_map_prompt(question, h)) if ok else ""
                    for h, ok in zip(hits, use, strict=True)]
        text = self.generator(build_reduce_prompt(question, findings))
        t2 = time.perf_counter()
        refused = is_no_evidence(text)
        return RAGResult(question, NO_EVIDENCE if refused else text, hits, relevance, findings,
                         retrieval_s=t1 - t0, generation_s=t2 - t1, refused=refused)


def format_sources(result: RAGResult, preview: int = 220) -> str:
    """Markdown list of the retrieved complaints for display."""
    if not result.hits:
        return "_No complaints retrieved._"
    lines = []
    for i, h in enumerate(result.hits, 1):
        snippet = h.text[:preview] + ("..." if len(h.text) > preview else "")
        rel = f" · relevance {result.relevance[i - 1]:.2f}" if i <= len(result.relevance) else ""
        finding = ""
        if result.findings:
            finding = (f"\n\n*Finding:* {result.findings[i - 1]}" if result.findings[i - 1]
                       else "\n\n*Not used: judged off-topic for this question.*")
        lines.append(
            f"**[{i}] Complaint {h.complaint_id}** · {h.product} · {h.issue} · "
            f"{h.company} · {h.date_received} · similarity {h.score:.2f}{rel}{finding}\n> {snippet}"
        )
    return "\n\n".join(lines)
