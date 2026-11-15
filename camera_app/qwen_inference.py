"""Classificador zero-shot afetivo com Qwen2.5-VL-3B-Instruct.

Reproduz a formulação do protocolo C do notebook experimental.
Não produz probabilidades calibradas nem diagnóstico clínico.
"""
import os
import re
from functools import lru_cache

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = os.getenv("FLORENCE_QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
PROMPT = (
    "Which category best describes this person's facial expression: "
    "pain, happiness, sadness, anger, disgust, or surprise? "
    "Answer with one category only."
)
CATEGORIES = ("pain", "happiness", "sadness", "anger", "disgust", "surprise")
DISPLAY = {
    "pain": "Person is in pain",
    "happiness": "Person is happy",
    "sadness": "Person is sad",
    "anger": "Person is angry",
    "disgust": "Person is disgusted",
    "surprise": "Person is surprised",
}

class QwenInferenceError(RuntimeError):
    pass

@lru_cache(maxsize=1)
def get_qwen():
    """Carregamento sob demanda. NF4 4-bit em CUDA, CPU em float32."""
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
    )
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, quantization_config=config, device_map="auto",
            torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to("cpu")
    return processor, model.eval()

def parse_category(answer):
    normalized = answer.strip().lower().strip(" .!\n\t\r:;\"'")
    if normalized in CATEGORIES:
        return normalized
    # Respostas fora do vocabulário não são convertidas silenciosamente em no-pain.
    raise QwenInferenceError(f"Resposta não categórica: {answer[:120]!r}")

def classify_expression(image):
    processor, model = get_qwen()
    image = image.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False, num_beams=1)
    generated = output[:, inputs.input_ids.shape[1]:]
    answer = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return DISPLAY[parse_category(answer)]
