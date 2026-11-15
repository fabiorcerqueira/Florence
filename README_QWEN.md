# Florence — Qwen2.5-VL-3B-Instruct Variant

This experimental variant replaces the original BLIP/LAVIS ITM facial-expression classification stage with **Qwen2.5-VL-3B-Instruct**. The BLIP VQA person-presence check (`Salesforce/blip-vqa-base`), Django web interface, camera capture, authentication workflow, and email-alert integration are retained.

## Requirements and installation

1. Create a Python 3.10+ virtual environment.
2. Install a PyTorch build compatible with your system and, if applicable, your CUDA version: https://pytorch.org/get-started/locally/.
3. Install the additional dependencies: `pip install -r requirements-qwen.txt`.
4. Set the configuration through environment variables (never commit secrets):
   - `FLORENCE_SECRET_KEY`: Django secret key. Without it, a random key is generated at each start and open sessions are lost on restart.
   - `FLORENCE_EMAIL_USER` and `FLORENCE_EMAIL_PASSWORD`: SMTP account used to send alerts (the default server is Gmail, `smtp.gmail.com:587`; use an app password).
   - `FLORENCE_ALERT_COOLDOWN_SECONDS` (optional, default `600`): minimum time between two alert e-mails for the same user.
   - `FLORENCE_DEBUG` (optional, default `1`) and `FLORENCE_ALLOWED_HOSTS` (optional, default `localhost,127.0.0.1`).
5. Create the database tables with `python manage.py migrate`. The app's migration is included in the repository.
6. Create a user with `python manage.py createsuperuser` and give it the e-mail address that should receive alerts. A user without an e-mail address receives no alerts; the camera page shows a warning when this happens.
7. Start the development server with `python manage.py runserver`, log in, and open the camera page.
8. Optional: run the automated tests, which replace the models with mocks and need no GPU or model download: `python manage.py test camera_app`.

If you already have a local `db.sqlite3` created with an earlier version (whose migrations were not versioned), either delete it and run the steps above, or run `python manage.py migrate camera_app --fake-initial`.

The first valid frame may trigger a model download and initialization. A CUDA-capable NVIDIA GPU is strongly recommended. On CUDA, the implementation uses bitsandbytes NF4 4-bit quantization; the CPU path uses float32 and may require substantial RAM and processing time. Start with one application worker to avoid loading multiple copies of the models.

## Classification protocol

The classifier uses a zero-shot prompt asking the model to select exactly one facial-expression category: `pain`, `happiness`, `sadness`, `anger`, `disgust`, or `surprise`. The prompt, image resolution, NF4 quantization, greedy decoding (20 new tokens) and strict parser are the same as in the benchmark notebook (protocol `C_affective`). Only the `pain` category activates the alert path.

An answer outside the expected category set is treated as **inconclusive**, rather than automatically interpreted as absence of pain. This implementation does not provide calibrated pain probabilities or a medical diagnosis.

## Configuration

The model checkpoint defaults to `Qwen/Qwen2.5-VL-3B-Instruct` and can be overridden through the `FLORENCE_QWEN_MODEL` environment variable.

## Monitoring behavior

- The camera page shows the time of the last analysis and the result of each frame: no person, the detected expression, an inconclusive answer, or a server error.
- If the login session expires, monitoring stops and the page asks the user to log in again, instead of silently discarding frames.
- After a pain result, the page shows whether the alert e-mail was sent, suppressed by the cooldown, not sent because the user has no e-mail address, or failed.
- "Stop capture" pauses monitoring; "Start capture" resumes it.

## Validation status

The parser reproduces the benchmark decisions for all 181 FPD-181 answers of protocol `C_affective`, and the web views are covered by automated tests with mocked models. **Full GPU inference, continuous camera operation, and email delivery have not yet been verified end to end.** Test these functions before considering deployment.

## Security and ethical considerations

Do not expose the Django development server to the public internet. Password changes require the logged-in user's current password, and frame uploads are CSRF-protected. Before any real-world use, review authentication, HTTPS, secret management, camera permissions, data retention, consent, and the legal rights to process or redistribute images. Human oversight remains essential.

## Related repositories

- Florence: https://github.com/fabiorcerqueira/Florence
- Image dataset and annotations: https://github.com/fabiorcerqueira/pain_nopain_img_rep
- Human-validation form generator: https://github.com/fabiorcerqueira/create_form_for_img_human_validation
