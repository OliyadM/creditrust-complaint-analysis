import gradio as gr
import sys
import os

# Add src/ to sys.path to import rag_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from rag_pipeline import rag_pipeline

# Define the Gradio interface function
def query_rag(question):
    """Process the user query and return the answer and sources."""
    if not question.strip():
        return "Please enter a question.", ""
    
    # Run the RAG pipeline
    try:
        answer, retrieved_chunks = rag_pipeline(question)
        
        # Format sources for display
        sources = "\n\n".join([
            f"**Complaint {chunk['complaint_id']} ({chunk['product']}):** {chunk['text'][:200]}..."
            for chunk in retrieved_chunks[:2]
        ])
        
        return answer, sources
    except Exception as e:
        return f"Error: {str(e)}", ""

# Clear the interface
def clear_inputs():
    """Reset the input and output fields."""
    return "", "", ""

# Create Gradio interface
with gr.Blocks(title="CrediTrust Complaint Analysis Chatbot") as demo:
    gr.Markdown("# CrediTrust Complaint Analysis Chatbot")
    gr.Markdown("Enter a question about customer complaints to get insights based on real feedback.")
    
    with gr.Row():
        question_input = gr.Textbox(label="Your Question", placeholder="e.g., Why are people unhappy with BNPL?")
        submit_button = gr.Button("Ask")
    
    answer_output = gr.Textbox(label="Answer", interactive=False)
    sources_output = gr.Textbox(label="Retrieved Sources (Top 2)", interactive=False)
    
    clear_button = gr.Button("Clear")
    
    # Connect buttons to functions
    submit_button.click(
        fn=query_rag,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    clear_button.click(
        fn=clear_inputs,
        inputs=None,
        outputs=[question_input, answer_output, sources_output]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()