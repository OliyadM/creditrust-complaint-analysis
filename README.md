Intelligent Complaint Analysis for Financial Services
Project Overview
This project develops a Retrieval-Augmented Generation (RAG) chatbot for CrediTrust Financial, a digital finance company in East Africa. The chatbot transforms unstructured customer complaint data into actionable insights, enabling product managers, support, and compliance teams to identify trends and issues across five product categories: Credit Card, Personal Loan, Buy Now, Pay Later (BNPL), Savings Account, and Money Transfers. Using the Consumer Financial Protection Bureau (CFPB) dataset, the system leverages semantic search (via FAISS) and a language model to answer plain-English questions, reducing analysis time from days to minutes.
This repository contains the code, data, and documentation for the interim submission, covering Task 1: Exploratory Data Analysis and Data Preprocessing and Task 2: Text Chunking, Embedding, and Vector Store Indexing.
Objectives

Task 1: Perform exploratory data analysis (EDA) and preprocess the CFPB dataset to filter for the five target products and clean complaint narratives.
Task 2: Implement text chunking, generate embeddings using sentence-transformers/all-MiniLM-L6-v2, and index them in a FAISS vector store.

Repository Structure
project_root/
├── data/
│   ├── filtered/
│   │   └── filtered_complaints.csv  # Cleaned and filtered CFPB dataset
├── notebooks/
│   └── eda_preprocessing.ipynb      # Jupyter Notebook for Task 1 (EDA and preprocessing)
├── src/
│   └── task2_chunk_embed_index.py   # Script for Task 2 (chunking, embedding, indexing)
├── vector_store/
│   ├── faiss_index                  # FAISS vector store index
│   └── metadata.pkl                 # Metadata for complaint chunks
├── reports/
│   └── interim_report.md            # Interim report summarizing Tasks 1 and 2
└── README.md                        # This file

Setup Instructions
To replicate the project environment and run the code, follow these steps:

Clone the Repository:
git clone <repository-url>
cd <repository-name>


Install Dependencies:Ensure Python 3.8+ is installed, then install required packages:
pip install pandas numpy matplotlib seaborn langchain sentence-transformers faiss-cpu


Download the Dataset:

Obtain the CFPB complaint dataset from the provided Dataset Link (replace with actual link if available).
Place the raw dataset in the data/ directory or update the paths in the scripts accordingly.


Run Task 1 (EDA and Preprocessing):

Open notebooks/eda_preprocessing.ipynb in Jupyter Notebook.
Execute the cells to perform EDA, filter the dataset, and save data/filtered/filtered_complaints.csv.


Run Task 2 (Chunking, Embedding, Indexing):
python src/task2_chunk_embed_index.py


This generates the FAISS index (vector_store/faiss_index) and metadata (vector_store/metadata.pkl).



Deliverables
Task 1: Exploratory Data Analysis and Data Preprocessing

Notebook: notebooks/eda_preprocessing.ipynb
Performs EDA on the CFPB dataset, analyzing complaint distribution, narrative length, and missing values.
Filters for five products and removes empty narratives.
Cleans narratives by lowercasing and removing special characters.


Filtered Dataset: data/filtered/filtered_complaints.csv
Contains 82,164 records for the target products with cleaned narratives.


Report Section: Included in reports/interim_report.md
Summarizes key findings, such as the dominance of Credit Card complaints and the median narrative length of 50 words.



Task 2: Text Chunking, Embedding, and Vector Store Indexing

Script: src/task2_chunk_embed_index.py
Chunks narratives using RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50).
Generates embeddings with sentence-transformers/all-MiniLM-L6-v2.
Indexes embeddings in a FAISS vector store with metadata (complaint ID, product category).


Vector Store: vector_store/faiss_index and vector_store/metadata.pkl
Stores embeddings and metadata for use in the RAG pipeline.


Report Section: Included in reports/interim_report.md
Details chunking strategy, embedding model choice, and FAISS implementation.



Interim Report

File: reports/interim_report.md
A Medium-style blog post summarizing Tasks 1 and 2, including business context, technical choices, findings, challenges, and next steps.



Submission Details

Interim Submission: Due 8:00 PM UTC, July 6, 2025
GitHub link to the main branch: <repository-url> (replace with actual link)
Includes all code, data, vector store files, and the interim report.



Next Steps

Task 3: Build the RAG pipeline with retriever, prompt template, and LLM integration, including qualitative evaluation.
Task 4: Develop a Gradio-based interactive chat interface for non-technical users.
Final Submission: Due 8:00 PM UTC, July 8, 2025

Team

Facilitators: Mahlet, Kerod, Rediet, Rehmet
Contributor: Oliyad Mulugeta

References

CFPB Dataset (replace with actual link)
LangChain Documentation
SentenceTransformers
FAISS Getting Started
Project guidelines and tutorials provided by facilitators.
