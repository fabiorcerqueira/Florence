"""Compatibilidade com a versão anterior (BLIP ITM).

A classificação agora é feita pelo Qwen2.5-VL-3B (ver qwen_inference.py).
Este módulo não é usado pelas views e não executa a checagem de presença.
"""
from io import BytesIO
from PIL import Image
from .qwen_inference import classify_expression


def process_frame_bytes(frame_bytes):
    image = Image.open(BytesIO(frame_bytes)).convert("RGB")
    category, sentiment = classify_expression(image)
    return {"detected": True, "sentiment": sentiment, "category": category}
