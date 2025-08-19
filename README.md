## Intelligent Complaint Analysis for   Financial Services
# Project Overview
This project delivers a Retrieval-Augmented Generation (RAG) chatbot for CrediTrust Financial, a digital finance company serving East African markets. The chatbot transforms unstructured customer complaint data from the Consumer Financial Protection Bureau (CFPB) dataset into actionable insights, empowering product managers, support, and compliance teams to address issues across five product categories: Credit Card, Personal Loan, Buy Now, Pay Later (BNPL), Savings Account, and Money Transfers. By leveraging semantic search (via FAISS) and a large language model, the system enables plain-English queries, reducing trend analysis time from days to minutes and fostering proactive issue resolution.
The repository contains all code, data, and documentation for the complete project, including data preprocessing, vector store indexing, RAG pipeline implementation, and a user-friendly Gradio interface.
Features

Data Processing: Filters and cleans CFPB complaint narratives for the five target products.
Semantic Search: Indexes complaint chunks in a FAISS vector store using sentence-transformers/all-MiniLM-L6-v2 for efficient retrieval.
RAG Pipeline: Retrieves relevant complaints and generates evidence-based answers using mistralai/Mixtral-8x7B-Instruct-v0.1.
Interactive Interface: A Gradio web interface allows non-technical users to query complaints and view answers with source complaints for transparency.

Repository Structure
project_root/
## data/
## ## filtered/
## ## ## filtered_complaints.csv  # Cleaned CFPB dataset
## notebooks/
## ## eda_preprocessing.ipynb      # EDA and preprocessing
## src/
## ## task2_chunk_embed_index.py   # Text chunking and indexing
## ## rag_pipeline.py             # RAG pipeline logic
## vector_store/
## ## faiss_index                 # FAISS vector store
## ## metadata.pkl                # Metadata for complaint chunks
## reports/
## ## final_report.md             # Comprehensive project report
## ## screenshots/               # Gradio interface screenshots
## app.py                        # Gradio interface
## README.md                     # This file

Setup Instructions
To run the project locally, follow these steps:
1. Clone the Repository
git clone <repository-url>
cd <repository-name>

2. Install Dependencies
Ensure Python 3.8+ is installed, then install required packages:
pip install pandas numpy matplotlib seaborn langchain sentence-transformers faiss-cpu transformers torch gradio

3. Dataset

Download the CFPB complaint dataset from CFPB Dataset (replace with actual link if available).
Place the raw dataset in data/ or update paths in scripts.

4. Run the Pipeline

Preprocessing: Open notebooks/eda_preprocessing.ipynb in Jupyter Notebook to filter and clean the dataset, saving results to data/filtered/filtered_complaints.csv.
Indexing: Run src/task2_chunk_embed_index.py to chunk narratives and create the FAISS index:python src/task2_chunk_embed_index.py


RAG Pipeline: Test the RAG pipeline with src/rag_pipeline.py:python src/rag_pipeline.py


Gradio Interface: Launch the web interface with app.py:python app.py

Access the interface at http://127.0.0.1:7860.

Note: The mistralai/Mixtral-8x7B-Instruct-v0.1 model in rag_pipeline.py is resource-intensive. For limited hardware, replace with distilbert-base-uncased by updating the model in rag_pipeline.py.
Deliverables
Data Preprocessing

Notebook: notebooks/eda_preprocessing.ipynb
Conducts exploratory data analysis (EDA) on complaint distribution, narrative length, and missing values.
Filters for 82,164 records across five products and cleans narratives (lowercase, remove special characters).


Output: data/filtered/filtered_complaints.csv

Text Chunking and Indexing

Script: src/task2_chunk_embed_index.py
Chunks narratives (chunk_size=500, chunk_overlap=50) using RecursiveCharacterTextSplitter.
Generates embeddings with sentence-transformers/all-MiniLM-L6-v2.
Indexes in FAISS with metadata (complaint ID, product).


Output: vector_store/faiss_index, vector_store/metadata.pkl

RAG Pipeline

Script: src/rag_pipeline.py
Retrieves top-5 complaint chunks using FAISS.
Generates answers with a prompt template and mistralai/Mixtral-8x7B-Instruct-v0.1.
Evaluates performance with 8 representative questions.


Evaluation: reports/final_report.md (Task 3 section)
Includes a table with questions, answers, sources, quality scores, and analysis.



Interactive Interface

Script: app.py
Gradio interface with text input, submit button, answer display, source display, and clear button.


Screenshots: reports/screenshots/ (includes initial state, query results, and cleared state).
Documentation: reports/final_report.md (Task 4 section)

Final Report

File: reports/final_report.md
A Medium-style blog post covering:
Business problem and RAG solution.
Technical choices (data, chunking, embeddings, LLM).
System evaluation with a table of test results.
UI showcase with screenshots.
Challenges, learnings, and future improvements.





Submission Details

Final Submission: Due 8:00 PM UTC, July 8, 2025
GitHub link: <repository-url> (replace with actual link)
Includes all code, data, vector store, reports, and screenshots.



Team

Facilitators: Mahlet, Kerod, Rediet, Rehmet
Contributor: Oliyad Mulugeta

We value collaboration and appreciate the guidance provided by the facilitators in building this project.
References

CFPB Dataset (replace with actual link)
LangChain Documentation
SentenceTransformers
FAISS Getting Started
Gradio Documentation
Hugging Face Transformers
Project guidelines and tutorials provided by facilitators.

Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub. Ensure code follows the project’s style and includes clear documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.