"""Small, hand-checked evaluation of the RAG pipeline.

  python -m src.evaluate run --model google/flan-t5-base
  python -m src.evaluate report

`run` answers every question in eval/questions.json and stores the raw
answers, retrieved sources, retrieval metrics and latency in
reports/eval_<model>.csv. Answer quality is then scored by hand (1-5, see the
rubric below) in eval/scores_<model>.csv, and `report` merges both into
reports/eval.md.

Retrieval metrics, for questions tied to a product:
  hit@k           at least one of the k retrieved complaints is that product
  precision@k     share of the k retrieved complaints that are that product
For the unanswerable questions the check is whether the pipeline declined.

Rubric (manual):
  5 correct, specific and fully supported by the shown complaints
  4 correct and supported, but generic or missing an obvious point
  3 partly answers; vague, or only one complaint really used
  2 mostly off-target or weakly supported
  1 wrong, unsupported, or refuses when the evidence was there
For unanswerable questions: 5 if it declines, 1 if it makes something up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import config

EVAL_DIR = config.ROOT / "eval"
QUESTIONS = EVAL_DIR / "questions.json"


def slug(model: str) -> str:
    return model.split("/")[-1]


def retrieval_metrics(expected: str | None, hit_products: list[str]) -> dict:
    if expected is None or not hit_products:
        return {"hit_at_k": None, "precision_at_k": None}
    matches = [p == expected for p in hit_products]
    return {"hit_at_k": float(any(matches)), "precision_at_k": sum(matches) / len(matches)}


def run(model: str, k: int = config.TOP_K) -> Path:
    from src.rag import RAGPipeline

    pipe = RAGPipeline.from_disk(gen_model=model)
    pipe.answer("warm up question about credit card fees", k=k)  # load weights/caches first
    rows = []
    for q in json.loads(QUESTIONS.read_text()):
        res = pipe.answer(q["question"], k=k)
        prods = [h.product for h in res.hits]
        rows.append({
            "id": q["id"],
            "expected_product": q["product"] or "(unanswerable)",
            "question": q["question"],
            "answer": res.answer,
            "findings": " || ".join(f for f in res.findings if f),
            "relevant": sum(1 for f in res.findings if f),
            "mean_relevance": round(sum(res.relevance) / len(res.relevance), 3) if res.relevance else None,
            "declined": res.refused,
            "top_score": round(res.hits[0].score, 3) if res.hits else None,
            "source_ids": " ".join(res.source_ids),
            "source_products": " | ".join(prods),
            **retrieval_metrics(q["product"], prods),
            "retrieval_s": round(res.retrieval_s, 3),
            "generation_s": round(res.generation_s, 3),
            "total_s": round(res.retrieval_s + res.generation_s, 3),
        })
        print(f"{q['id']} [{rows[-1]['total_s']:.1f}s] {res.answer[:160]}")
    out = config.REPORTS_DIR / f"eval_{slug(model)}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def summarise(raw: pd.DataFrame, scores: pd.DataFrame | None) -> dict:
    answerable = raw[raw["expected_product"] != "(unanswerable)"]
    unans = raw[raw["expected_product"] == "(unanswerable)"]
    s = {
        "questions": len(raw),
        "hit_at_k": answerable["hit_at_k"].mean(),
        "precision_at_k": answerable["precision_at_k"].mean(),
        "answerable_declined": int(answerable["declined"].sum()),
        "unanswerable_declined": f"{int(unans['declined'].sum())}/{len(unans)}",
        "median_latency_s": raw["total_s"].median(),
        "max_latency_s": raw["total_s"].max(),
    }
    if scores is not None:
        s["mean_score"] = scores["score"].mean()
        s["mean_score_answerable"] = scores[scores["id"].isin(answerable["id"])]["score"].mean()
    return s


def report() -> Path:
    raw_files = sorted(f for f in config.REPORTS_DIR.glob("eval_*.csv") if f.stem != "eval_summary")
    md = ["# Evaluation", "",
          "12 hand-written questions (10 tied to a product, 2 unanswerable), top-k = 5,",
          "run on CPU (8 cores). Scores are assigned by hand using the rubric in",
          "`src/evaluate.py`. `relevant` is how many of the 5 retrieved complaints",
          "passed the relevance check and were used for the answer.", ""]
    summary_rows = []
    sections = []
    for f in raw_files:
        model = f.stem.removeprefix("eval_")
        raw = pd.read_csv(f)
        score_file = EVAL_DIR / f"scores_{model}.csv"
        scores = pd.read_csv(score_file) if score_file.exists() else None
        s = summarise(raw, scores)
        summary_rows.append({"generator": model, **s})
        table = raw[["id", "expected_product", "hit_at_k", "precision_at_k", "relevant", "total_s",
                     "answer"]].copy()
        if scores is not None:
            table = table.merge(scores, on="id", how="left")
        table["answer"] = table["answer"].str.replace("|", "/", regex=False).str.replace("\n", " ")
        sections += [f"## {model}", "", table.to_markdown(index=False, floatfmt=".2f"), ""]
    md += ["## Summary", "", pd.DataFrame(summary_rows).to_markdown(index=False, floatfmt=".2f"), ""]
    md += sections
    out = config.REPORTS_DIR / "eval.md"
    out.write_text("\n".join(md))
    pd.DataFrame(summary_rows).to_csv(config.REPORTS_DIR / "eval_summary.csv", index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default=config.GEN_MODEL)
    r.add_argument("--k", type=int, default=config.TOP_K)
    sub.add_parser("report")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        print(run(args.model, args.k))
    else:
        print(report())


if __name__ == "__main__":
    main()
