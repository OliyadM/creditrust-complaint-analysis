import pandas as pd
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pickle
import os

# Set paths
DATA_PATH = "data/filtered/filtered_complaints.csv"
VECTOR_STORE_PATH = "vector_store/faiss_index"
METADATA_PATH = "vector_store/metadata.pkl"

# Step 1: Load the cleaned dataset
def load_data():
    """Load the filtered complaints dataset."""
    df = pd.read_csv(DATA_PATH)
    return df

# Step 2: Text Chunking
def chunk_text(texts, chunk_size=500, chunk_overlap=50):
    """Split texts into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = []
    metadata = []
    for idx, text in enumerate(texts):
        text_chunks = text_splitter.split_text(text)
        chunks.extend(text_chunks)
        metadata.extend([{"original_index": idx, "chunk_index": i} for i in range(len(text_chunks))])
    return chunks, metadata

# Step 3: Generate Embeddings
def generate_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Generate embeddings for text chunks using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)
    return embeddings

# Step 4: Create and Save FAISS Index
def create_faiss_index(embeddings, metadata):
    """Create a FAISS index and save it along with metadata."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Create vector_store directory if it doesn't exist
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, VECTOR_STORE_PATH)
    
    # Save metadata
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    
    return index

# Main execution
def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Extract narratives and metadata
    texts = df['cleaned_narrative'].tolist()
    metadata = [{
        "complaint_id": row['Complaint ID'],
        "product": row['Product'],
        "original_index": idx
    } for idx, row in df.iterrows()]
    
    # Chunk texts
    print("Chunking texts...")
    chunks, chunk_metadata = chunk_text(texts)
    
    # Update metadata with additional information
    for i, meta in enumerate(chunk_metadata):
        original_idx = meta['original_index']
        meta.update({
            "complaint_id": metadata[original_idx]['complaint_id'],
            "product": metadata[original_idx]['product']
        })
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    
    # Create and save FAISS index
    print("Creating FAISS index...")
    create_faiss_index(embeddings, chunk_metadata)
    
    print(f"Vector store saved to {VECTOR_STORE_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")

if __name__ == "__main__":
    main()