import chromadb
import fitz  # type: ignore
from numpy import ndarray

from rag_pipeline.vector import (convert_image_to_embeddings,
                                 convert_text_to_embeddings)

from .config import PDF_PATH, VECTOR_DB_DIR


def extract_text_and_images_from_pdf() -> tuple:
    doc = fitz.open(PDF_PATH)

    texts = []
    images = []

    for page in doc:
        texts.append(page.get_text())
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            filename = f"image_{page.number}_{xref}.png"
            with open(filename, "wb") as f:
                f.write(image_bytes)
            images.append(filename)

    text_embeddings = convert_text_to_embeddings(texts)
    image_embeddings = convert_image_to_embeddings(images)
    return text_embeddings, image_embeddings


def store_embeddings(text_embeddings: ndarray, image_embeddings: list):
    client = chromadb.PersistentClient(VECTOR_DB_DIR)
    collection = client.get_or_create_collection("documents")
    for i, embedding in enumerate(text_embeddings):
        collection.upsert(ids=[f"Text {i}"],
                          documents=[f"Text {i}"],
                          embeddings=[embedding.tolist()],
                          metadatas=[{
                              "type": "text",
                              "index": i
                          }])
    for i, embedding in enumerate(image_embeddings):
        collection.upsert(ids=[f"image_{i}"],
                          embeddings=[embedding.squeeze().tolist()],
                          metadatas=[{
                              "type": "image",
                              "index": i
                          }])
    return collection
