---
title: CrediTrust Complaint Assistant
emoji: 💬
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.13"
app_file: app.py
pinned: false
---

RAG assistant over CFPB consumer complaint narratives for five product lines
(credit cards, personal loans, BNPL, savings accounts, money transfers).
Source: https://github.com/OliyadM/creditrust-complaint-analysis

Set the Space variable `CREDITRUST_INDEX_REPO` to a dataset repo holding
`index.faiss`, `chunks.parquet` and `info.json` to load the full index.
Without it the Space builds a small demo index from one CFPB archive file
on first start (a few minutes on CPU basic).
