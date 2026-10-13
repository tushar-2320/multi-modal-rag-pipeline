from numpy import ndarray
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPVisionModelWithProjection


def convert_text_to_embeddings(texts: list) -> ndarray:
    text_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    text_embeddings = text_model.encode(texts)
    return text_embeddings


def convert_image_to_embeddings(images: list) -> list:
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch16")
    image_vectors: list = []
    for image_path in images:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        features = outputs.image_embeds
        image_vectors.append(features.cpu().detach().numpy())
    return image_vectors
