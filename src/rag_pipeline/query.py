import chromadb

from rag_pipeline.vector import convert_text_to_embeddings

from .config import VECTOR_DB_DIR


def query_documents(query: str, top_k: int = 5):
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collection = client.get_collection("documents")

    query_embedding = convert_text_to_embeddings([query])[0]

    results = collection.query(query_embeddings=[query_embedding.tolist()],
                               n_results=top_k,
                               where={"type": "text"},
                               include=["documents", "metadatas", "distances"])

    return results
