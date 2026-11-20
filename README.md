# Florence — Vision-Language Framework for Assistive Pain Monitoring

Florence is an experimental, Python-based system designed to support the monitoring of people who may require continuous care. It processes camera frames using vision-language models to identify whether a person is present and to classify visible facial expressions associated with pain.

Florence is intended for research and assistive monitoring. **A facial-expression classification is not a clinical diagnosis or a reliable measure of an individual's pain experience.** Outputs should not replace human assessment or professional care.

## System overview

The application uses Django for its web interface. The browser captures a camera frame at a configurable interval (default 30 s) and sends it to the server, which decodes it (OpenCV/Pillow) and runs the pipeline:

1. **Person-presence check:** a BLIP visual-question-answering model (`Salesforce/blip-vqa-base`).
2. **Facial-expression classification:** the current Qwen variant uses `Qwen/Qwen2.5-VL-3B-Instruct` in a zero-shot, six-category protocol: `pain`, `happiness`, `sadness`, `anger`, `disgust`, and `surprise`.
3. **Monitoring and notification:** the page displays each result; when the model returns `pain`, the server sends an e-mail to the logged-in user's registered address, at most once per cooldown period (default 10 min). An answer outside the six categories is treated as inconclusive: no decision and no alert.

The Qwen variant replaces the earlier BLIP/LAVIS image–text matching (ITM) expression classifier; it does **not** replace the BLIP VQA presence check.

## Research and reproducibility

The Florence research project includes a human-annotated image collection and experimental comparisons of vision-language architectures, prompting strategies, and statistical analyses. The current paper evaluation uses **FPD-181 (181 images with binary pain/no-pain reference labels)**; this should not be confused with the 189-image collection used in earlier experiments, which included eight inconclusive cases.

- **Application and experimental code:** https://github.com/fabiorcerqueira/Florence
- **Human annotations:** image-level labels and all human ratings are in the reproducibility package below. The photographs (from free stock-photo websites) are not redistributed because of third-party licenses; requests can be addressed to the authors, subject to those licenses.
- **Human-validation form generation:** https://github.com/fabiorcerqueira/create_form_for_img_human_validation
- **Experimental notebook:** [`testing_models.ipynb`](testing_models.ipynb). It was run in Google Colab, and its setup cell clones an image repository that is no longer public. Rerunning the inference requires the images, placed under `/content/florence_visapp/images/` (`PainImages/` and `NoPainImages/`). The statistics can be reproduced without the images from the reproducibility package.

The reproducibility package is [`Florence_supplement_for_reproducibility.zip`](Florence_supplement_for_reproducibility.zip): image-level predictions, image-level labels, all human ratings, statistical scripts and documentation.

## Running the Qwen variant

See [`README_QWEN.md`](README_QWEN.md) for installation requirements, configuration, and limitations. A CUDA-capable NVIDIA GPU is recommended. The Qwen-enabled application has **not yet been end-to-end validated with live GPU inference**; syntax checks alone do not establish operational reliability.

## Responsible use

Use the system only with appropriate consent and safeguards. Review privacy, image permissions, email configuration, access controls, and deployment security before use with real people. The system is a research prototype, not a certified medical device.

## Citation

If you use Florence in research, please cite the corresponding paper once its bibliographic details are available. Do not cite an unpublished venue or DOI as if it were already published.
