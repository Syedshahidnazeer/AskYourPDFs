import streamlit as st
import streamlit.components.v1 as components
import json
import os
from src.rag_pipeline import RAGPipeline
import yaml
import logging

@st.cache_data # Use st.memo to prevent re-initialization
def initialize_pipeline():
    return RAGPipeline()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Ensure assets and styles directories exist
os.makedirs('assets', exist_ok=True)
os.makedirs('styles', exist_ok=True)

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize RAG Pipeline
rag_pipeline = RAGPipeline()

def local_css(file_name):
    """Load local CSS file"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_lottie_file(filename):
    """Load Lottie JSON animation file"""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback Lottie JSON if file not found
        return {
            "v": "5.5.7",
            "fr": 29.9700012207031,
            "ip": 0,
            "op": 180.00000733155,
            "w": 512,
            "h": 512,
            "nm": "Loader",
            "ddd": 0,
            "assets": [],
            "layers": []
        }

def render_lottie_animation(lottie_json, element_id, width=200, height=200):
    """Render Lottie animation in Streamlit"""
    components.html(f'''
    <script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.7.14/lottie.min.js"></script>
    <div id="{element_id}" style="width: {width}px; height: {height}px;"></div>
    <script>
        var animation = lottie.loadAnimation({{
            container: document.getElementById('{element_id}'),
            renderer: 'svg',
            loop: true,
            autoplay: true,
            animationData: {json.dumps(lottie_json)}
        }});
    </script>
    ''', height=height, width=width)

def main():
    # Page Configuration
    st.set_page_config(
        page_title="PDF RAG Assistant", 
        page_icon=":books:", 
        layout="wide"
    )
    st.markdown(
    f'<link rel="stylesheet" href="static/styles.css">',
    unsafe_allow_html=True,
)
    # Load Custom CSS (create a styles/main.css file with your custom styles)
    local_css("styles/main.css")
    
    # Title Section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🧠 PDF Insight Generator")
    with col2:
        # Render PDF Upload Lottie Animation
        pdf_lottie = load_lottie_file("assets/pdf-upload.json")
        render_lottie_animation(pdf_lottie, "pdf-lottie", 100, 100)
    
    # PDF Upload
    st.subheader("📤 Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type=['pdf'], 
        accept_multiple_files=True,
        help="Upload multiple PDFs for comprehensive analysis"
    )

    rag_pipeline = initialize_pipeline()
    # Processing Indicator
    if uploaded_files:
        if not hasattr(rag_pipeline, 'documents'): # Checks if already processed
            with st.spinner('Processing Documents...'):
                rag_pipeline.process_documents(uploaded_files)
            st.success(f"Processed {len(uploaded_files)} documents successfully!")
        
        # Search Section
        st.subheader("🔍 Ask Questions About Your Documents")
        query = st.text_input(
            "Enter your query", 
            placeholder="What insights do you want from these PDFs?"
        )
        
        if query:
                with st.spinner('Generating Intelligent Response...'):
                    response = rag_pipeline.retrieve_and_generate(query)  # Generate response first

                # Display Response FIRST
                st.subheader("📄 Response")
                st.write(response)


                # NOW render the animations (move these AFTER the response)
                result_lottie = load_lottie_file("assets/result-animation.json")
                render_lottie_animation(result_lottie, "result-lottie", 100, 100)

if __name__ == "__main__":
    main()