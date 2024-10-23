import os
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
import json
import re

from search_docs import index_documents, semantic_search, generate_answer, load_lottieurl

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    p = Path(__file__).parent
    metadata_file = p / 'metadata.json'
    doc_index_file = p / 'document_index.faiss'

    print("Starting the application...")

    model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
    dimension = 768
    index = faiss.IndexFlatIP(dimension)
    metadata = []

    print(f"Initialized model and FAISS index with dimension {dimension}")
    print("Starting Streamlit UI")

    if not os.path.exists(doc_index_file):
        p = Path(__file__).parent.parent
        documents_path = p / 'assets'
        index_documents(documents_path, model, metadata, index, metadata_file)

    if len(metadata) == 0:
        print("Loading FAISS index and metadata")
        index = faiss.read_index(str(doc_index_file))
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        print(f"Loaded index with {index.ntotal} vectors and {len(metadata)} metadata entries")

    question = 'what was the fingo profit for 2022'

    if question:
        print(f"User asked: '{question}'")

        search_results = semantic_search(question, model, metadata, index)
        context = "\n\n".join([f"{i}: {result['content']}" for i, result in enumerate(search_results)])

        answer = generate_answer(question, context)

        print(f'ANSWER:\n{answer}')
        rege = re.compile(r"\[Document\s+[0-9]+\]|\[[0-9]+\]")
        referenced_ids = [int(s) for s in re.findall(r'\b\d+\b', ' '.join(rege.findall(answer)))]

        print(f"Displaying {len(referenced_ids)} referenced documents")
        for doc_id in referenced_ids:
            doc = search_results[doc_id]
            print(f"📄 Document {doc_id} - {os.path.basename(doc['path'])}")
            #     st.write(doc['content'])
            #     with open(doc['path'], 'rb') as f:
            #         st.download_button("⬇️ Download file", f, file_name=os.path.basename(doc['path']))
    # else:
    #     st.warning("⚠️ Please enter a question before clicking 'Search and Answer'.")


if __name__ == "__main__":
    main()
    print("Application finished")
