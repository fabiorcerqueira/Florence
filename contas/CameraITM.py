import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from lavis.models import load_model_and_preprocess

import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BLIP VQA
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model_vqa = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

# LAVIS ITM
model_itm, vis_processors_itm, text_processors_itm = load_model_and_preprocess(
    name="blip2_image_text_matching",
    model_type="coco",  # ou "flickr"
    is_eval=True,
    device=device
)

frases = [
    'Person is in pain',
    'Person is angry',
    'Person is disgusted',
    'Person is happy',
    'Person is sad',
    'Person is surprised'
]
textos_itm = [text_processors_itm["eval"](frase) for frase in frases]


def process_frame_bytes(frame_bytes):
    """
    Recebe bytes de imagem (JPEG/PNG) e retorna o resultado da análise
    """
    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

    # VQA: pergunta se há pessoa
    question = "Is there a person in the image?"
    inputs = processor(img, question, return_tensors="pt").to(device)
    out = model_vqa.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)

    result = {"detected": False, "sentiment": None}

    if answer.lower() == "yes":
        # ITM: detectar sentimento
        img_itm = vis_processors_itm["eval"](img).unsqueeze(0).to(device)
        scores = []
        for txt in textos_itm:
            with torch.no_grad():
                itm_output = model_itm({"image": img_itm, "text_input": txt}, match_head="itm")
                itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
                scores.append(itm_scores[:, 1].item())

        maior = max(scores)
        h = scores.index(maior)
        sentimento_detectado = frases[h]

        result["detected"] = True
        result["sentiment"] = sentimento_detectado

    return result
