"""Compatibilidade: a classificação é realizada pelo Qwen2.5-VL-3B."""
from io import BytesIO
from PIL import Image
from .qwen_inference import classify_expression

def process_frame_bytes(frame_bytes):
    image = Image.open(BytesIO(frame_bytes)).convert("RGB")
    return {"detected": True, "sentiment": classify_expression(image)}
