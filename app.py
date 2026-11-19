"""Gradio front end for the CrediTrust complaint assistant.

  python app.py            # http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import os

import gradio as gr

from src import config
from src.bootstrap import ensure_index
from src.products import PRODUCTS
from src.rag import RAGPipeline, format_sources

ALL = "All products"

index_dir = ensure_index()
pipeline = RAGPipeline.from_disk(index_dir)
info = json.loads((index_dir / "info.json").read_text())

EXAMPLES = [
    "Why are customers unhappy with buy now pay later services?",
    "What problems do people report with credit card late fees?",
    "Why are international money transfers delayed?",
    "What issues come up when withdrawing from a savings account?",
]


def ask(question: str, product: str, k: int):
    if not question or not question.strip():
        return "Please type a question.", ""
    res = pipeline.answer(question, k=int(k), product=None if product == ALL else product)
    answer = res.answer
    if not res.refused:
        answer += "\n\n**Sources:** " + ", ".join(res.source_ids)
    answer += f"\n\n_retrieval {res.retrieval_s:.2f}s · generation {res.generation_s:.1f}s_"
    return answer, format_sources(res)


with gr.Blocks(title="CrediTrust complaint assistant") as demo:
    gr.Markdown(
        "# CrediTrust complaint assistant\n"
        "Ask about customer complaints for credit cards, personal loans, BNPL, savings "
        "accounts and money transfers. Answers are generated only from the retrieved "
        "CFPB complaint narratives listed below them.\n\n"
        f"Index: {info['complaints']:,} complaints / {info['chunks']:,} chunks "
        f"({info.get('source', '')}). Generator: `{config.GEN_MODEL}` on CPU."
    )
    with gr.Row():
        question = gr.Textbox(label="Question", lines=2, scale=4)
        with gr.Column(scale=1):
            product = gr.Dropdown([ALL] + PRODUCTS, value=ALL, label="Product filter")
            k = gr.Slider(3, 8, value=config.TOP_K, step=1, label="Complaints to retrieve")
    with gr.Row():
        submit = gr.Button("Ask", variant="primary")
        clear = gr.Button("Clear")
    answer = gr.Markdown(label="Answer")
    sources = gr.Markdown(label="Retrieved complaints")
    gr.Examples(EXAMPLES, inputs=question)

    submit.click(ask, [question, product, k], [answer, sources], api_name="ask")
    question.submit(ask, [question, product, k], [answer, sources], api_name=False)
    clear.click(lambda: ("", ALL, config.TOP_K, "", ""), None, [question, product, k, answer, sources],
                api_name=False)

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name=os.environ.get("HOST", "0.0.0.0"),
                                   server_port=int(os.environ.get("PORT", "7860")))
