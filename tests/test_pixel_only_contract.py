"""Guard the AVCS pixel-vision inference path against shortcut dependencies."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIXEL_VISION_PATHS = (
    Path("forge") / "vision",
    Path("avcs") / "goal_grounder.py",
    Path("avcs") / "human_eye.py",
    Path("avcs") / "controller.py",
    Path("avcs") / "focus.py",
    Path("avcs") / "verifier.py",
)
FORBIDDEN_IMPORT_ROOTS = {
    "easyocr",
    "paddleocr",
    "playwright",
    "pyautogui",
    "pytesseract",
    "pywinauto",
    "selenium",
    "uiautomation",
}


def _python_files(path: Path):
    absolute = ROOT / path
    assert absolute.exists(), f"Pixel-vision source path does not exist: {path}"
    if absolute.is_file():
        yield absolute
        return
    yield from absolute.rglob("*.py")


def _imported_modules(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_pixel_vision_inference_does_not_import_external_ocr_or_ui_automation():
    violations = []
    for source_path in PIXEL_VISION_PATHS:
        for file_path in _python_files(source_path):
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for module in _imported_modules(tree):
                root_module = module.split(".", maxsplit=1)[0]
                if root_module in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(
                        f"{file_path.relative_to(ROOT)} imports {module}"
                    )

    assert not violations, (
        "Pixel-vision inference must use local visual-model evidence, not OCR, "
        "DOM/browser automation, or accessibility shortcuts:\n"
        + "\n".join(violations)
    )
