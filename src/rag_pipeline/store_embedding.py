from rag_pipeline.extract import (extract_text_and_images_from_pdf,
                                  store_embeddings)


def main():
    print("Extracting text and images from PDF...")

    text_embeddings, image_embeddings = (extract_text_and_images_from_pdf())

    print(f"Text embeddings: {len(text_embeddings)}")
    print(f"Image embeddings: {len(image_embeddings)}")

    print("Storing embeddings in ChromaDB...")

    collection = store_embeddings(text_embeddings, image_embeddings)

    print(f"Total stored embeddings: {collection.count()}")
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
