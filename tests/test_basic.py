"""
Basic smoke tests for the STAT GR5293 research agent repo.

Run:
    python test_basic.py

These tests avoid API calls and do not launch Gradio.
"""

import importlib.util
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "demo.ipynb"
OCR_HELPER = ROOT / "src" / "extract_pdf_text_with_ocr.py"


def check(condition, message):
    if condition:
        print(f"[PASS] {message}")
        return True
    print(f"[FAIL] {message}")
    return False


def check_import(module_name):
    try:
        __import__(module_name)
        print(f"[PASS] import {module_name}")
        return True
    except Exception as e:
        print(f"[FAIL] import {module_name}: {e}")
        return False


def check_notebook_clean():
    if not check(NOTEBOOK.exists(), "demo.ipynb exists"):
        return False

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    outputs = sum(1 for cell in nb.get("cells", []) if cell.get("outputs"))
    exec_counts = sum(
        1
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
        and cell.get("execution_count") is not None
    )

    ok = True
    ok &= check(outputs == 0, "notebook has no saved outputs")
    ok &= check(exec_counts == 0, "notebook has no execution counts")

    text = NOTEBOOK.read_text(encoding="utf-8")
    ok &= check("sk-" not in text, "notebook does not appear to contain OpenAI API keys")
    return ok


def check_project_files():
    ok = True
    ok &= check((ROOT / "requirements.txt").exists(), "requirements.txt exists")
    ok &= check(
        (ROOT / "requirements-evaluation.txt").exists(),
        "requirements-evaluation.txt exists",
    )
    ok &= check((ROOT / ".gitignore").exists(), ".gitignore exists")
    ok &= check(OCR_HELPER.exists(), "OCR helper file exists")
    return ok


def check_ocr_helper_import():
    helper = OCR_HELPER
    if not helper.exists():
        print("[FAIL] extract_pdf_text_with_ocr.py not found")
        return False

    spec = importlib.util.spec_from_file_location("extract_pdf_text_with_ocr", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    has_function = hasattr(module, "extract_text_with_ocr")
    return check(has_function, "OCR helper exposes extract_text_with_ocr")


def check_optional_ocr_tools():
    tesseract = shutil.which("tesseract")
    pdftoppm = shutil.which("pdftoppm")

    if tesseract and pdftoppm:
        print(f"[PASS] tesseract found at {tesseract}")
        print(f"[PASS] pdftoppm found at {pdftoppm}")
        return True

    print("[WARN] OCR system tools not found. This is okay unless testing scanned PDFs.")
    print("       macOS: brew install poppler tesseract")
    print("       Colab/Linux: apt-get install -y poppler-utils tesseract-ocr")
    return True


def main():
    print("Running basic repository smoke tests...\n")

    ok = True

    ok &= check_project_files()
    ok &= check_notebook_clean()

    print("\nChecking package imports...")
    for module in [
        "requests",
        "wikipediaapi",
        "PyPDF2",
        "dotenv",
        "nltk",
        "torch",
        "transformers",
        "sentence_transformers",
        "langchain_openai",
        "gradio",
    ]:
        ok &= check_import(module)

    print("\nChecking OCR helper...")
    ok &= check_ocr_helper_import()
    ok &= check_optional_ocr_tools()

    print("\nResult:")
    if ok:
        print("[PASS] Basic smoke tests passed.")
    else:
        print("[FAIL] Some smoke tests failed.")


if __name__ == "__main__":
    main()
