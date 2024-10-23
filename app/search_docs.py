import json
import os
from pathlib import Path

import faiss
import numpy as np
import ollama
import requests

from read_docs import read_pdf, read_docx, read_pptx, chunk_text, read_document_chunk


def index_documents(directory: Path, model, metadata, index, metadata_file):
    print(f"Indexing documents in directory: {directory}")

    documents = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")
            content = ""

            if file.endswith('.pdf'):
                content = read_pdf(file_path)
            elif file.endswith('.docx'):
                content = read_docx(file_path)
            elif file.endswith('.pptx'):
                content = read_pptx(file_path)
            elif file.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                raise ValueError(file)

            if content:
                chunks = chunk_text(content)
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadata.append({"path": file_path, "chunk_id": i})

    print(f"Encoding {len(documents)} document chunks")
    embeddings = model.encode(documents)
    print(f"Adding embeddings to FAISS index")
    index.add(np.array(embeddings))

    # Save index and metadata
    print("Saving FAISS index and metadata")
    faiss.write_index(index, "document_index.faiss")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    print(f"Indexed {len(documents)} document chunks.")


def semantic_search(query, model, metadata, index, k=10):
    print(f"Performing semantic search for query: '{query}', k={k}")
    query_vector = model.encode([query])[0]
    distances, indices = index.search(np.array([query_vector]), k)

    results = []
    for i, idx in enumerate(indices[0]):
        meta = metadata[idx]
        content = read_document_chunk(meta["path"], meta["chunk_id"])
        results.append({
            "id": int(idx),
            "path": meta["path"],
            "content": content,
            "score": float(distances[0][i])
        })

    print(f"Found {len(results)} search results")
    return results


def generate_answer(query, context):
    # llm_model = 'Bielik-11B-v2.2-Instruct.Q8_0.gguf'
    llm_model = 'tinyllama'

    print(f"Generating answer for query: '{query}'")
#     prompt = f"""Answer the user's question using the documents given in the context.
#     In the context are documents that should contain an answer.
#     Please always reference the document ID (in square brackets, for example [0],[1]) of the document that
#     was used to make a claim. Use as many citations and documents as it is necessary to answer the question.
#
# Context:
# {context}
#
# Question: {query}
#
# Answer:"""

    prompt = f"""Odpowiedz na pytanie uzytkownika korzystając z tekstu zawartego w kontekście. 
        W kontekscie są dokumenty, ktore powinny zawierać odpowiedź. 
        Zawsze podawaj identyfikator dokumentu (w nawiasach kwadratowych, np. [0],[1]).
        Użyj jak najwięcej cytatów z dokumentu.

    Kontekst:
    {context}

    Pytanie: {query}

    odpowiedź:"""

    #
    print(f"PROMPT:{prompt}")
    ollama.pull(llm_model)
    response = ollama.generate(model=llm_model, prompt=prompt)
    # print("Received response from Ollama")
    return response['response']


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
