import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.prompts import PromptTemplate

# Paths
VECTOR_STORE_PATH = "../vector_store/faiss_index"
METADATA_PATH = "../vector_store/metadata.pkl"
DATA_PATH = "../data/filtered/filtered_complaints.csv"

# Step 1: Load Vector Store and Metadata
def load_vector_store():
    """Load FAISS index and metadata."""
    index = faiss.read_index(VECTOR_STORE_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

# Step 2: Retrieve Relevant Chunks
def retrieve_chunks(query, index, metadata, model, top_k=5):
    """Embed query and retrieve top-k chunks from FAISS."""
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    retrieved_chunks = []
    for idx in indices[0]:
        chunk_info = metadata[idx]
        chunk_info['distance'] = distances[0][list(indices[0]).index(idx)]
        retrieved_chunks.append(chunk_info)
    return retrieved_chunks

# Step 3: Generate Answer
def generate_answer(query, retrieved_chunks, llm_pipeline):
    """Generate answer using retrieved chunks and LLM."""
    # Prepare context from retrieved chunks
    context = "\n".join([f"Complaint {chunk['complaint_id']} ({chunk['product']}): {chunk['text']}" for chunk in retrieved_chunks])
    
    # Define prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints based solely on the provided context. If the context doesn't contain enough information to answer the question, state that clearly. Provide a concise, evidence-based answer.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )
    
    # Format prompt
    prompt = prompt_template.format(context=context, question=query)
    
    # Generate answer
    response = llm_pipeline(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]['generated_text']
    
    # Extract answer part (remove prompt from response)
    answer_start = response.find("Answer:") + len("Answer:")
    answer = response[answer_start:].strip()
    
    return answer, retrieved_chunks

# Main RAG Pipeline
def rag_pipeline(query):
    """Run the full RAG pipeline for a given query."""
    # Initialize models
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm_pipeline = pipeline(
        "text-generation",
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        device=-1,  # Use CPU; change to 0 for GPU if available
        model_kwargs={"load_in_4bit": True}  # Optimize for memory
    )
    
    # Load vector store
    index, metadata = load_vector_store()
    
    # Retrieve chunks
    retrieved_chunks = retrieve_chunks(query, index, metadata, embedder)
    
    # Load original texts for context
    df = pd.read_csv(DATA_PATH)
    for chunk in retrieved_chunks:
        original_idx = chunk['original_index']
        chunk['text'] = df.iloc[original_idx]['cleaned_narrative']
    
    # Generate answer
    answer, retrieved_chunks = generate_answer(query, retrieved_chunks, llm_pipeline)
    
    return answer, retrieved_chunks

# Example Usage
if __name__ == "__main__":
    import pandas as pd
    
    query = "Why are people unhappy with BNPL?"
    answer, retrieved_chunks = rag_pipeline(query)
    
    print("Question:", query)
    print("Answer:", answer)
    print("\nRetrieved Sources:")
    for i, chunk in enumerate(retrieved_chunks[:2], 1):
        print(f"Source {i}: Complaint {chunk['complaint_id']} ({chunk['product']}): {chunk['text'][:100]}...")