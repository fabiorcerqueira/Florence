# Florence — Qwen2.5-VL-3B-Instruct Variant

This experimental variant replaces the original BLIP/LAVIS ITM facial-expression classification stage with **Qwen2.5-VL-3B-Instruct**. The BLIP VQA person-presence check (`Salesforce/blip-vqa-base`), Django web interface, camera capture, authentication workflow, and email-alert integration are retained.

## Requirements and installation

1. Create a Python 3.10+ virtual environment.
2. Install a PyTorch build compatible with your system and, if applicable, your CUDA version: https://pytorch.org/get-started/locally/.
3. Install the additional dependencies: `pip install -r requirements-qwen.txt`.
4. Configure Django settings and email credentials securely in `camera_project/settings.py`; do not commit passwords, tokens, or secret keys.
5. Run database migrations with `python manage.py migrate`.
6. Start the development server with `python manage.py runserver`, log in, and test camera capture.

The first valid frame may trigger a model download and initialization. A CUDA-capable NVIDIA GPU is strongly recommended. On CUDA, the implementation uses bitsandbytes NF4 4-bit quantization; the CPU path uses float32 and may require substantial RAM and processing time. Start with one application worker to avoid loading multiple copies of the models.

## Classification protocol

The classifier uses a zero-shot prompt asking the model to select exactly one facial-expression category: `pain`, `happiness`, `sadness`, `anger`, `disgust`, or `surprise`. The model output is parsed into the application's existing `detected`/`sentiment` response format. Only the `pain` category activates the existing pain-alert path.

An answer outside the expected category set is treated as **inconclusive**, rather than automatically interpreted as absence of pain. This implementation does not provide calibrated pain probabilities or a medical diagnosis.

## Configuration

The model checkpoint defaults to `Qwen/Qwen2.5-VL-3B-Instruct` and can be overridden through the `FLORENCE_QWEN_MODEL` environment variable.

## Validation status

Python syntax and basic parsing checks were performed when this variant was prepared. **Full GPU inference, continuous camera operation, and email-alert behavior have not yet been verified end to end.** Test these functions before considering deployment.

## Security and ethical considerations

Do not expose the Django development server to the public internet. Before any real-world use, review authentication, CSRF protection, secret management, camera permissions, data retention, consent, and the legal rights to process or redistribute images. Human oversight remains essential.

## Related repositories

- Florence: https://github.com/fabiorcerqueira/Florence
- Image dataset and annotations: https://github.com/fabiorcerqueira/pain_nopain_img_rep
- Human-validation form generator: https://github.com/fabiorcerqueira/create_form_for_img_human_validation
