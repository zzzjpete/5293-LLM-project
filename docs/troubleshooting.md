# Troubleshooting

This document lists common problems and fixes for running the Research Assistant Agent project.

## Missing API Keys

If the notebook reports missing credentials, make sure these variables are set:

```text
OPENAI_API_KEY
SERPER_API_KEY
HF_TOKEN
```

In Google Colab, add them through the Secrets manager.

For local runs, create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key
HF_TOKEN=your_huggingface_token
```

## Package Import Errors

If Python cannot import a required package, reinstall the dependencies:

```bash
pip install -r requirements.txt
```

For evaluation notebooks, also run:

```bash
pip install -r requirements-evaluation.txt
```

If you use a virtual environment, make sure it is activated before installing packages or running notebooks.

## Tests Fail

Run the smoke test from the project root:

```bash
python tests/test_basic.py
```

If import checks fail, install dependencies first:

```bash
pip install -r requirements.txt
```

If using a local virtual environment, run the test with that environment's Python executable.

## OCR Not Working

OCR requires system tools in addition to Python packages.

On macOS:

```bash
brew install poppler tesseract
```

On Colab or Linux:

```bash
apt-get install -y poppler-utils tesseract-ocr
```

The project includes an OCR helper:

```text
src/extract_pdf_text_with_ocr.py
```

Example:

```bash
python src/extract_pdf_text_with_ocr.py test_cases/scanned_pdf_test_document.pdf
```

## Scanned PDF Has No Extractable Text

Some scanned PDFs contain only images, not embedded text. Normal PDF extraction may return empty output for these files.

Use OCR extraction for scanned PDFs. OCR quality depends on scan resolution, page layout, and image quality.

## Gradio Demo Does Not Launch

Make sure Gradio is installed:

```bash
pip install -r requirements.txt
```

If running in Colab, use the public share link printed by Gradio.

If running locally, open the local URL printed by Gradio, usually:

```text
http://127.0.0.1:7860
```

## Web Search Fails

The web search tool uses Serper. Check that:

- `SERPER_API_KEY` is set
- The key is valid
- The network connection is working
- The Serper API service is available

## OpenAI Calls Fail

Check that:

- `OPENAI_API_KEY` is set
- The key has available credits
- The requested model is available
- The network connection is working

## Hugging Face Model Downloads Fail

Some embedding and reranking models are downloaded from Hugging Face.

If downloads fail, check that:

- The network connection is working
- `HF_TOKEN` is set if the model requires authentication
- The runtime has enough disk space and memory

## Results Differ From README

Small result differences are expected because the project uses:

- Live web search
- LLM API calls
- External model downloads
- External datasets
- Changing search results

The saved notebooks and plots document the experimental setup and representative results.
