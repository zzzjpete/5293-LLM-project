"""
Checks that the demo upload files are present and usable.

Run from the project root:
    python tests/test_demo_inputs.py
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_CASES = ROOT / "test_cases"

EXPECTED_FILES = [
    "400-motivational-quotes.pdf",
    "Untitled document_copy.pdf",
    "sample_upload_test_document.txt",
    "scanned_pdf_test_document.pdf",
]


def check(condition, message):
    if condition:
        print(f"[PASS] {message}")
        return True
    print(f"[FAIL] {message}")
    return False


def check_expected_files():
    ok = check(TEST_CASES.exists(), "test_cases directory exists")
    ok &= check(TEST_CASES.is_dir(), "test_cases is a directory")

    for filename in EXPECTED_FILES:
        path = TEST_CASES / filename
        ok &= check(path.exists(), f"{filename} exists")
        if path.exists():
            ok &= check(path.stat().st_size > 0, f"{filename} is not empty")

    return ok


def check_pdf_headers():
    ok = True
    for path in TEST_CASES.glob("*.pdf"):
        header = path.read_bytes()[:5]
        ok &= check(header == b"%PDF-", f"{path.name} has a PDF header")
    return ok


def check_text_inputs():
    ok = True
    for path in TEST_CASES.glob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        ok &= check(bool(text), f"{path.name} contains text")
    return ok


def main():
    print("Checking demo input files...\n")

    ok = True
    ok &= check_expected_files()
    ok &= check_pdf_headers()
    ok &= check_text_inputs()

    print("\nResult:")
    if ok:
        print("[PASS] Demo input checks passed.")
        return 0

    print("[FAIL] Demo input checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
