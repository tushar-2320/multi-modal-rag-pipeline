# RAG Pipeline

A lightweight multi-modal retrieval pipeline for PDF documents. It extracts text and images from a PDF, converts them into embeddings, stores them in a persistent ChromaDB collection, and retrieves relevant chunks for text-based queries.

## Features

- PDF text extraction with PyMuPDF
- Page image extraction from PDF documents
- Text embeddings using SentenceTransformers
- Image embeddings using CLIP
- Persistent vector storage with ChromaDB
- Query interface for asking questions about the indexed PDF

## Project structure

```text
src/
  rag_pipeline/
    __init__.py
    config.py
    extract.py
    vector.py
    main.py
    query.py
```

- `config.py`: environment configuration and model settings
- `extract.py`: PDF parsing and embedding storage logic
- `vector.py`: text and image embedding generation
- `main.py`: end-to-end extraction and indexing pipeline
- `query.py`: interactive retrieval script

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Access to the PDF you want to index

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Create a `.envrc` file with the required environment variables. This project loads it with `dotenv`.

```bash
cat > .envrc <<'EOF'
PDF_PATH=/absolute/path/to/your/document.pdf
VECTOR_DB_DIR=vector_db
TEXT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
IMAGE_EMBEDDING_MODEL=openai/clip-vit-base-patch16
EOF
```

The relevant configuration values are:

- `PDF_PATH`: full path to the PDF file
- `VECTOR_DB_DIR`: folder where ChromaDB persists the vector database
- `TEXT_EMBEDDING_MODEL`: embedding model used for text chunks
- `IMAGE_EMBEDDING_MODEL`: CLIP vision model used for extracted images

## Run the indexing pipeline

```bash
uv run python -m rag_pipeline.main
```

This command:

- opens the PDF
- extracts page text and images
- generates text and image embeddings
- stores all vectors in ChromaDB under the `documents` collection

## Query the indexed PDF

```bash
uv run python -m rag_pipeline.query
```

The script prompts for a question and then performs a similarity search against the ChromaDB collection. It returns the top matching document chunks and metadata.

## Notes

- The vector database is stored persistently in the folder configured by `VECTOR_DB_DIR`.
- Extracted images are saved to the working directory as `image_*.png` files before embedding.
- Retrieval is currently focused on text-based queries; image embeddings are stored alongside text embeddings, but the query script uses text embeddings for search.
