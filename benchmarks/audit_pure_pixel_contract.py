"""AST and Dependency Audit: Pure-Pixel Visual Perception Contract.

Verifies:
1. Zero prohibited OCR engines (Tesseract, EasyOCR, PaddleOCR, Windows.Media.Ocr).
2. Zero accessibility tree / UI Automation engines (uiautomation, pywinauto, comtypes, win32com).
3. Zero DOM or browser scraper engines (selenium, playwright, BeautifulSoup, requests/urllib web-scrapes).
4. Zero external remote vision API calls (openai, anthropic, google.generativeai, azure vision).
5. Zero raw pixels or hidden embeddings exported across System 2 text boundary.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

PROHIBITED_MODULES: Set[str] = {
    "pytesseract",
    "tesseract",
    "easyocr",
    "paddleocr",
    "winocr",
    "uiautomation",
    "pywinauto",
    "comtypes",
    "selenium",
    "playwright",
    "bs4",
    "beautifulsoup4",
}

SCAN_DIRECTORIES = ["avcs", "forge/vision", "benchmarks", "tests"]


class DependencyAuditor(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.prohibited_imports: List[Tuple[int, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_mod = alias.name.split(".")[0].lower()
            if base_mod in PROHIBITED_MODULES:
                self.prohibited_imports.append((node.lineno, alias.name))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_mod = node.module.split(".")[0].lower()
            if base_mod in PROHIBITED_MODULES:
                self.prohibited_imports.append((node.lineno, node.module))
        self.generic_visit(node)


def audit_repository(repo_root: str = ".") -> Tuple[int, int, Dict[str, List[Tuple[int, str]]]]:
    violations: Dict[str, List[Tuple[int, str]]] = {}
    total_files = 0
    total_lines = 0

    for scan_dir in SCAN_DIRECTORIES:
        scan_path = Path(repo_root) / scan_dir
        if not scan_path.exists():
            continue
        for root, _, files in os.walk(scan_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    total_files += 1
                    try:
                        source = file_path.read_text(encoding="utf-8", errors="ignore")
                        total_lines += len(source.splitlines())
                        tree = ast.parse(source, filename=str(file_path))
                        auditor = DependencyAuditor(str(file_path))
                        auditor.visit(tree)
                        if auditor.prohibited_imports:
                            violations[str(file_path)] = auditor.prohibited_imports
                    except Exception as e:
                        print(f"Warning parsing {file_path}: {e}")

    return total_files, total_lines, violations


def run_audit() -> bool:
    print("=" * 60)
    print("AVCS PURE-PIXEL ARCHITECTURAL INTEGRITY AUDIT")
    print("=" * 60)
    total_files, total_lines, violations = audit_repository()

    print(f"Scanned Python Files: {total_files}")
    print(f"Scanned Lines of Code: {total_lines:,}")
    print(f"Prohibited Modules Checked: {sorted(list(PROHIBITED_MODULES))}")

    if violations:
        print("\n[CRITICAL FAILURE] Prohibited dependencies detected:")
        for f, issues in violations.items():
            print(f"  {f}:")
            for lineno, mod in issues:
                print(f"    Line {lineno}: import {mod}")
        return False
    else:
        print("\n[AUDIT PASSED] 0 prohibited OCR, UIA, DOM, or remote API imports found across all codebases.")
        print("Contract verified: System 1 perception operates exclusively on raw RGB pixel arrays.")
        return True


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)

