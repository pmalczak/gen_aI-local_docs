import os
import faiss
from sentence_transformers import SentenceTransformer
import json
import streamlit as st
import re
from streamlit_lottie import st_lottie

from search_docs import index_documents, semantic_search, generate_answer, load_lottieurl

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    print("Starting the application...")

    # Global variables
    model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
    dimension = 768
    index = faiss.IndexFlatIP(dimension)
    metadata = []

    print(f"Initialized model and FAISS index with dimension {dimension}")

    print("Starting Streamlit UI")
    
    # Page config
    st.set_page_config(page_title="Local GenAI Search", page_icon="🔍", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #1E90FF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 12px;
        border: none;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and animation
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<p class="big-font">Local GenAI Search 🔍</p>', unsafe_allow_html=True)
        st.write("Explore your documents with the power of AI!")
    with col2:
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json"
        lottie_json = load_lottieurl(lottie_url)
        st_lottie(lottie_json, height=150, key="coding")

    # Input for documents path
    documents_path = st.text_input("📁 Enter the path to your documents folder:", "Folder Path")
    
    # Check if documents are indexed
    if not os.path.exists("document_index.faiss"):
        st.warning("⚠️ Documents are not indexed. Please run the indexing process first.")
        if st.button("🚀 Index Documents"):
            with st.spinner("Indexing documents... This may take a while."):
                print(f"Indexing documents in {documents_path}")
                index_documents(documents_path, model)
            st.success("✅ Indexing complete!")
            st.experimental_rerun()  # Rerun the app after indexing

    # Load index and metadata if not already loaded
    global index, metadata
    if len(metadata) == 0:
        print("Loading FAISS index and metadata")
        index = faiss.read_index("document_index.faiss")
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
        print(f"Loaded index with {index.ntotal} vectors and {len(metadata)} metadata entries")
    
    st.markdown("---")
    st.markdown("## Ask a Question")
    question = st.text_input("🤔 What would you like to know about your documents?", "")

    if st.button("🔍 Search and Answer"):
        if question:
            with st.spinner("Searching and generating answer..."):
                print(f"User asked: '{question}'")
                
                # Perform semantic search
                search_results = semantic_search(question, model)
                
                # Prepare context for answer generation
                context = "\n\n".join([f"{i}: {result['content']}" for i, result in enumerate(search_results)])
                
                # Generate answer
                answer = generate_answer(question, context)
                
                st.markdown("### 🤖 AI Answer:")
                st.markdown(answer)
                
                # Display referenced documents
                st.markdown("### 📚 Referenced Documents:")
                rege = re.compile(r"\[Document\s+[0-9]+\]|\[[0-9]+\]")
                referenced_ids = [int(s) for s in re.findall(r'\b\d+\b', ' '.join(rege.findall(answer)))]
                
                print(f"Displaying {len(referenced_ids)} referenced documents")
                for doc_id in referenced_ids:
                    doc = search_results[doc_id]
                    with st.expander(f"📄 Document {doc_id} - {os.path.basename(doc['path'])}"):
                        st.write(doc['content'])
                        with open(doc['path'], 'rb') as f:
                            st.download_button("⬇️ Download file", f, file_name=os.path.basename(doc['path']))
        else:
            st.warning("⚠️ Please enter a question before clicking 'Search and Answer'.")


if __name__ == "__main__":
    main()
    print("Application finished")