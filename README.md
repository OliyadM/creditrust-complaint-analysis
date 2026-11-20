# CrediTrust complaint analysis (RAG over CFPB complaints)

CrediTrust Financial's product, support and compliance teams want to ask plain-English
questions such as *"why are customers unhappy with BNPL?"* and get an answer backed by real
customer complaints instead of reading thousands of them. This project builds that tool on the
public CFPB Consumer Complaint Database for CrediTrust's five product lines: **credit cards,
personal loans, Buy Now Pay Later (BNPL), savings accounts and money transfers**.

Everything runs locally on CPU with free models: MiniLM embeddings, a FAISS index and
`google/flan-t5-base` for generation, behind a small Gradio app.

## What changed in v2

v1 (Jul–Aug 2025, the history below this rebuild) was a course project. Revisiting it in
Nov 2026, several things did not hold up, so v2 rebuilds the pipeline on top of it:

- **Counts were not reproducible.** v1 filtered on exact names like `"Personal loan"` and
  `"Buy Now, Pay Later (BNPL)"`, which are not CFPB product values, so only credit cards (and the
  legacy money-transfer label) actually matched. The 82,164 figure in the old README could not be
  reproduced and is dropped. v2 maps every historical CFPB product/sub-product name and detects
  BNPL by company and keywords (`src/products.py`), with per-product counts above.
- **Data source changed.** CFPB removed narratives from the live bulk download in 2026; v2 reads
  the official narratives archive with a script (`src/ingest.py`) instead of a local CSV.
- **New pipeline.** Chunk text is stored with the index (v1 re-read the full CSV by row index),
  generation uses `flan-t5-base` on CPU instead of Mixtral-8x7B (not runnable on a normal machine),
  answers go through a relevance check and per-complaint summaries, and imports/paths work from
  the repo root.
- **Real evaluation.** 12 hand-written questions with retrieval metrics, latency and manually
  scored answers (`reports/eval.md`), instead of no measured results.
- **Tests, CI and complete requirements.** pytest suite, GitHub Actions lint + tests, and pinned
  requirements (v1 listed only pandas/numpy/matplotlib/seaborn/jupyter).
- The v1 notebook and scripts (`notebooks/eda and preprocessing.ipynb`, `src/chunk_embed_index.py`,
  `src/rag_pipeline.py`) are replaced by `notebooks/eda.ipynb` and the modules in `src/`; they
  remain in the git history.

## Data

**Source.** CFPB removed narratives from the live bulk download, so the data comes from the
[CFPB Consumer Complaint Database Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/):
21 zip files covering complaints received Dec 2011 – Aug 2026 (about 1.4 GB zipped).
No data is stored in this repo.

```bash
python -m src.ingest download   # 21 zips -> data/raw/
python -m src.ingest filter     # -> data/filtered/filtered_complaints.csv.gz  (~9 min)
python -m src.ingest sample --per-product 4000   # -> data/processed/sample_complaints.csv.gz
```

**Product mapping** (`src/products.py`). CFPB renamed its products several times, so each
CrediTrust product is matched against every historical name:

| CrediTrust product | CFPB product / sub-product used |
|---|---|
| Credit card | `Credit card` (legacy); `Credit card or prepaid card` → general-purpose or store credit card only (prepaid excluded) |
| Personal loan | `Consumer Loan`, `Payday loan, title loan, or personal loan`, `Payday loan, title loan, personal loan, or advance loan` → installment loan or personal line of credit only (payday, title, pawn excluded) |
| Savings account | `Checking or savings account` / `Bank account or service` → savings account or CD only |
| Money transfers | `Money transfers` (legacy); `Money transfer, virtual currency, or money service` → domestic, international, mobile/digital wallet |
| BNPL | not a CFPB product. Company is Affirm, Klarna, Sezzle, Zip, Afterpay or Splitit, **or** the narrative mentions buy now pay later / BNPL / Klarna / Afterpay / Sezzle / Pay in 4 / Quadpay. Takes priority over the other labels. |

**Counts** (rows with a non-empty narrative, deduplicated by complaint ID):

| | complaints |
|---|---:|
| Rows read from the archive | 17,533,918 |
| Rows with a narrative | 3,851,415 (22.0%) |
| **Credit card** | **227,147** |
| **Personal loan** | **25,330** |
| **BNPL** | **8,537** |
| **Savings account** | **18,685** |
| **Money transfers** | **104,670** |
| **Total in scope** | **384,369** (narratives from 2015-03-19 to 2026-07-29) |

An earlier version of this project quoted 82,164 filtered complaints; that number came from an
exact-name filter that only matched credit cards and could not be reproduced, so it is not used.

**Sample for the index.** Embedding all 384k narratives on a CPU takes hours, so the vector store
is built from a **seeded stratified sample of 4,000 complaints per product (20,000 total, seed 42)**.
Equal group sizes keep BNPL from being drowned out by credit cards, but it means the index does not
reflect real complaint volumes; do not use it to compare how often products are complained about.

EDA (product mix, years, narrative length, top issues, cleaning examples): `notebooks/eda.ipynb`.

## How it works

```
CFPB archive zips ─► src/ingest.py ─► filtered CSV ─► stratified sample
                                                         │ clean (drop XXXX masks, unwrap {$amounts})
                                                         ▼
                            src/chunking.py  RecursiveCharacterTextSplitter 500 / 50
                                                         │ 20,000 complaints -> 62,543 chunks
                                                         ▼
                 src/vector_store.py  all-MiniLM-L6-v2 (384-d, normalised) -> FAISS IndexFlatIP
                                                         │ chunk text + complaint_id, product, issue, company, date
question ─► embed ─► top-5 chunks (one per complaint, optional product filter)
                          │
                          ├─ best cosine < 0.40 ............................ decline
                          ├─ flan-t5 P("yes") "is this complaint related?" per chunk
                          │     mean < 0.30 ................................. decline
                          ├─ map: one-sentence finding per relevant complaint
                          └─ reduce: two-sentence answer from the findings only
                                                         ▼
                  answer + IDs of the complaints it was written from (src/rag.py, app.py)
```

Why map/reduce: with all five excerpts in one prompt, flan-t5 mostly copied one excerpt or
answered with a bare citation. Summarising one complaint at a time and then summarising the
findings gave clearly better answers. Why the relevance check: retrieval similarity alone could
not separate on-topic from off-topic questions (both land around 0.6), so the generator is asked
whether each retrieved complaint relates to the question and the probability of "yes" is used.
All thresholds were chosen on the separate dev questions in `eval/dev_questions.json`.

Complaint IDs listed with an answer are attached by the pipeline (they are the complaints whose
findings fed the summary); the model does not write citations itself.

## Results

12 hand-written questions (`eval/questions.json`): two per product plus two unanswerable ones,
top-k = 5, 8-core CPU. Answer quality is scored by hand 1–5 with the rubric in
`src/evaluate.py`; every score has a one-line justification in `reports/eval.md`.

| generator | hit@5 | product precision@5 | answerable declined | unanswerable declined | median / max latency | mean score (all 12) | mean score (10 answerable) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **flan-t5-base** (default) | 1.00 | 0.90 | 1 / 10 | 2 / 2 | 1.6 s / 3.4 s | 3.08 | 2.70 |
| flan-t5-large | 1.00 | 0.90 | 3 / 10 | 2 / 2 | 3.4 s / 9.7 s | 3.08 | 2.70 |

What this says:

- **Retrieval is the strong part.** For every answerable question at least one of the top 5
  complaints was the right product, and 90% of retrieved complaints were. Misses were mostly
  adjacent products (a BNPL refund question pulling a credit card dispute).
- **Generation is the weak part.** Answers are short, often restate the question, or copy one
  complaint word for word. Typical score is 3 ("partly answers"). The per-complaint findings shown
  in the app are often more useful than the summary.
- **Declining works, but is too eager.** Both unanswerable questions were declined, but the
  relevance check also declined one answerable question with base (q01) and three with large;
  the thresholds were tuned on base and not re-tuned for large.
- flan-t5-large writes somewhat more specific answers when it does answer, but at 2–3× the latency
  and with more false declines it is not worth it here, so base is the default.

Full table with answers, sources and justifications: [`reports/eval.md`](reports/eval.md),
raw runs in `reports/eval_flan-t5-*.csv`.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # CPU torch; on macOS drop the +cpu pin
python -m src.ingest download && python -m src.ingest filter && python -m src.ingest sample
python -m src.vector_store                 # ~4 min on 8 cores
python app.py                              # http://127.0.0.1:7860
python -m src.evaluate run --model google/flan-t5-base && python -m src.evaluate report
```

If `vector_store/` is empty, `app.py` will first try a prebuilt index from a Hugging Face dataset
(`CREDITRUST_INDEX_REPO=<user>/<dataset>`), and otherwise build a small demo index from one archive
file (Jul–Aug 2025, 500 complaints per product, 7,917 chunks; about 1 minute on 8 cores).

Tests and lint (no models or data needed; this is what CI runs):

```bash
pip install -r requirements-dev.txt
ruff check . && pytest -q
```

### Sample output

From `reports/app_smoke_test.md` (app running locally, queried through `gradio_client`):

```text
Q: Why are customers unhappy with buy now pay later services? (3.7s)
A: They make over $2000.00 in service charges alone. They take the money before they process
   the order. They don't pay off the full purchase over 12 months. They are allowing my merchants
   to get paid without delivering products.
   Sources: 8186981, 10377333, 9128942, 6409880, 6276615

Q: What problems do people report with credit card late fees?  [filter: Credit card] (1.5s)
A: The late fees hit their credit score and credit report.
   Sources: 1574480, 4339924, 5742736

Q: Who won the football world cup? (0.1s)
A: Not enough evidence in the retrieved complaints to answer this.
```

The BNPL answer shows both sides: the points are grounded in the cited complaints, but they are
stitched together in the customers' first-person voice rather than summarised.

## Deployment (Hugging Face Spaces, free CPU)

1. Build the index locally, then upload it to a dataset repo:
   `HF_TOKEN=... python -m src.publish_index --repo <user>/creditrust-complaint-index`
2. Create a Gradio Space and push `app.py`, `src/`, `requirements.txt`, using
   `deploy/huggingface/README.md` as the Space's README (it holds the Space config).
3. In the Space settings set the variable `CREDITRUST_INDEX_REPO=<user>/creditrust-complaint-index`.
   Without it the Space builds the demo index on first start.

The index is about 100 MB (FAISS vectors + chunk text) and the models are downloaded from the Hub
at startup. Expect slower answers on the 2-vCPU free tier than the timings above.

## Limitations

- The index covers a 20k sample, not all 384k complaints, and is balanced by design.
- BNPL is keyword/company based: it misses BNPL complaints that never name a provider and can
  include complaints that only mention one in passing (some are filed under credit reporting or
  debt collection).
- Answer quality is limited by a 250M-parameter model; summaries are often generic or copied.
- The evaluation is small (12 questions) and scored by the author, so treat the scores as a sanity
  check, not a benchmark. Thresholds were tuned on 23 dev questions; they will misfire on some
  phrasings.
- CFPB narratives are consumer-submitted and unverified; CFPB masks personal details with `XXXX`,
  which sometimes leaves sentences with gaps.

## Repo layout

```
app.py                     Gradio app
src/config.py              paths and defaults
src/products.py            CFPB -> CrediTrust product mapping, narrative cleaning
src/ingest.py              download, filter, stratified sample
src/chunking.py            chunking with metadata
src/vector_store.py        embeddings + FAISS build/load
src/rag.py                 retriever, relevance check, map/reduce generation
src/evaluate.py            evaluation runner and report
src/bootstrap.py           index loading for the app (local / Hub / demo build)
src/publish_index.py       upload index to a Hugging Face dataset
notebooks/eda.ipynb        exploratory analysis (executed)
eval/                      questions, dev questions, manual scores
reports/                   counts, evaluation tables, app smoke test
tests/                     pytest suite (tiny fixtures, no models)
deploy/huggingface/        Space config
```
