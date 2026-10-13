from numpy import ndarray
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

from rag_pipeline.config import IMAGE_EMBEDDING_MODEL, TEXT_EMBEDDING_MODEL


def convert_text_to_embeddings(texts: list) -> ndarray:
    text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    text_embeddings = text_model.encode(texts)
    return text_embeddings


def convert_image_to_embeddings(images: list) -> list:
    processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
    model = CLIPVisionModelWithProjection.from_pretrained(
        IMAGE_EMBEDDING_MODEL)
    image_vectors: list = []
    for image_path in images:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        features = outputs.image_embeds
        image_vectors.append(features.cpu().detach().numpy())
    return image_vectors
