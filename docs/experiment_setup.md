# Experiment Setup

This document explains how to reproduce the main demo and evaluation experiments for the Research Assistant Agent project.

## Environment

The project was developed primarily in Google Colab with Python 3.12. It can also be run locally with a virtual environment.

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Install the extra evaluation dependencies:

```bash
pip install -r requirements-evaluation.txt
```

## Required API Keys

The notebooks use the following environment variables:

```text
OPENAI_API_KEY
SERPER_API_KEY
HF_TOKEN
```

`OPENAI_API_KEY` is used for LLM calls.

`SERPER_API_KEY` is used for web search through Serper.

`HF_TOKEN` is optional unless Hugging Face model access is required.

In Google Colab, add these keys with the Secrets manager. For local runs, place them in a `.env` file in the project root.

## Main Notebooks

The project has two main notebooks:

```text
notebooks/demo.ipynb
notebooks/evaluation.ipynb
```

Use `notebooks/demo.ipynb` for the agent demo, tool behavior, document upload workflow, and Gradio UI.

Use `notebooks/evaluation.ipynb` for evaluation experiments, benchmark runs, ablation studies, statistical tests, and generated plots.

## Demo Reproduction

To reproduce the demo:

1. Open `notebooks/demo.ipynb`.
2. Install dependencies from `requirements.txt`.
3. Add the required API keys.
4. Run the setup and tool-definition cells.
5. Run the agent implementation cells.
6. Launch the Gradio interface.
7. Upload files from `test_cases/` to test document loading, OCR behavior, quote retrieval, and summarization.

The `test_cases/` folder contains sample PDFs, scanned PDFs, and text files that can be used to test the UI.

## Evaluation Reproduction

To reproduce the evaluation results:

1. Install both dependency files:

```bash
pip install -r requirements.txt
pip install -r requirements-evaluation.txt
```

2. Set the required API keys.
3. Open `notebooks/evaluation.ipynb`.
4. Run the setup cells.
5. Run the curated ReAct vs. CoT benchmark.
6. Run the ablation study.
7. Run the HotpotQA evaluation section if runtime allows.
8. Run the plotting cells.

Generated figures should be saved in `results/`.

## Evaluation Metrics

The project reports metrics such as:

- Tool Selection F1
- Keyword Coverage
- Answer Similarity
- Task Completion Rate
- Runtime
- Number of tool steps
- Exact Match
- Token F1
- Quote found rate
- Verified quote found rate

## Reproducibility Notes

Exact outputs may vary across runs because the project uses live web search, external APIs, model calls, and external datasets.

Important sources of variation include:

- Search result changes
- API availability
- LLM response variability
- Hugging Face model downloads
- Colab runtime limits
- Network speed and availability

The notebooks, requirements files, sample test cases, and saved plots are included to make the experiments as reproducible as possible.
