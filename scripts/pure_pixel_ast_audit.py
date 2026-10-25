"""Comprehensive AST and Code Integrity Auditor for AVCS Pure-Pixel Contract.

Checks both splendid-babbage and avcs-sdk for:
1. Prohibited OCR engines: pytesseract, tesseract, easyocr, paddleocr, winocr
2. Prohibited UI Automation / DOM: uiautomation, pywinauto, selenium, playwright, bs4, beautifulsoup4, comtypes, win32com
3. Decoupling of PixelWordRecognizer and character n-gram vocabularies in the perception loop.
"""
from __future__ import annotations

import ast
import json
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
    "selenium",
    "playwright",
    "bs4",
    "beautifulsoup4",
    "comtypes",
    "win32com",
}

REPOS = {
    "splendid-babbage": Path(r"c:\Users\nandk\Documents\antigravity\splendid-babbage"),
    "avcs-sdk": Path(r"c:\Users\nandk\Documents\antigravity\avcs-sdk"),
}

EXCLUDE_DIRS = {
    ".venv",
    ".git",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
    "egg-info",
    "avcs_sdk.egg-info",
}

class ASTSecurityVisitor(ast.NodeVisitor):
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.violations: List[Tuple[int, str, str]] = []
        self.pwr_references: List[Tuple[int, str]] = []
        self.ngram_references: List[Tuple[int, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_mod = alias.name.split(".")[0].lower()
            if base_mod in PROHIBITED_MODULES:
                self.violations.append((node.lineno, f"import {alias.name}", base_mod))
            if "pixelwordrecognizer" in alias.name.lower():
                self.pwr_references.append((node.lineno, f"import {alias.name}"))
            if "charngram" in alias.name.lower():
                self.ngram_references.append((node.lineno, f"import {alias.name}"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_mod = node.module.split(".")[0].lower()
            if base_mod in PROHIBITED_MODULES:
                self.violations.append((node.lineno, f"from {node.module} import ...", base_mod))
            if "pixelwordrecognizer" in node.module.lower():
                self.pwr_references.append((node.lineno, f"from {node.module} import ..."))
            if "charngram" in node.module.lower():
                self.ngram_references.append((node.lineno, f"from {node.module} import ..."))

        for alias in node.names:
            if alias.name.lower() in PROHIBITED_MODULES:
                self.violations.append((node.lineno, f"import {alias.name}", alias.name))
            if "pixelwordrecognizer" in alias.name.lower():
                self.pwr_references.append((node.lineno, f"import {alias.name}"))
            if "charngram" in alias.name.lower():
                self.ngram_references.append((node.lineno, f"import {alias.name}"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check call to __import__ or importlib.import_module
        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                val = node.args[0].value.lower()
                if val.split(".")[0] in PROHIBITED_MODULES:
                    self.violations.append((node.lineno, f"__import__('{node.args[0].value}')", val))
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                val = node.args[0].value.lower()
                if val.split(".")[0] in PROHIBITED_MODULES:
                    self.violations.append((node.lineno, f"import_module('{node.args[0].value}')", val))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "PixelWordRecognizer":
            self.pwr_references.append((node.lineno, f"Name: {node.id}"))
        elif "CharNGram" in node.id:
            self.ngram_references.append((node.lineno, f"Name: {node.id}"))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "PixelWordRecognizer":
            self.pwr_references.append((node.lineno, f"Attribute: {node.attr}"))
        elif "CharNGram" in node.attr:
            self.ngram_references.append((node.lineno, f"Attribute: {node.attr}"))
        self.generic_visit(node)


def audit_all():
    report = {
        "summary": {},
        "violations": {},
        "pwr_references": {},
        "ngram_references": {},
    }

    total_files = 0
    total_lines = 0

    for repo_name, repo_path in REPOS.items():
        repo_files = 0
        repo_lines = 0
        print(f"Auditing repository: {repo_name} ({repo_path})")
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not any(part in d for part in ["egg-info"])]
            for file in files:
                if file.endswith(".py"):
                    filepath = Path(root) / file
                    rel_path = f"{repo_name}/{filepath.relative_to(repo_path)}"
                    total_files += 1
                    repo_files += 1
                    try:
                        content = filepath.read_text(encoding="utf-8", errors="ignore")
                        lines = len(content.splitlines())
                        total_lines += lines
                        repo_lines += lines
                        tree = ast.parse(content, filename=str(filepath))
                        visitor = ASTSecurityVisitor(filepath)
                        visitor.visit(tree)

                        if visitor.violations:
                            report["violations"][rel_path] = visitor.violations
                        if visitor.pwr_references:
                            report["pwr_references"][rel_path] = visitor.pwr_references
                        if visitor.ngram_references:
                            report["ngram_references"][rel_path] = visitor.ngram_references
                    except Exception as e:
                        print(f"Error parsing {filepath}: {e}")

        report["summary"][repo_name] = {
            "files": repo_files,
            "lines": repo_lines,
        }

    report["summary"]["total_files"] = total_files
    report["summary"]["total_lines"] = total_lines
    return report

if __name__ == "__main__":
    res = audit_all()
    print("\n--- AUDIT SUMMARY ---")
    print(json.dumps(res["summary"], indent=2))
    print("\n--- PROHIBITED IMPORT / CALL VIOLATIONS ---")
    print(f"Total files with violations: {len(res['violations'])}")
    for f, v in res["violations"].items():
        print(f"{f}: {v}")
    
    print("\n--- PIXEL WORD RECOGNIZER REFERENCES ---")
    print(f"Files referencing PixelWordRecognizer: {len(res['pwr_references'])}")
    for f, v in res["pwr_references"].items():
        print(f"{f}: {len(v)} occurrences (e.g. line {v[0][0]}: {v[0][1]})")

    print("\n--- CHAR N-GRAM REFERENCES ---")
    print(f"Files referencing CharNGram: {len(res['ngram_references'])}")
    for f, v in res["ngram_references"].items():
        print(f"{f}: {len(v)} occurrences (e.g. line {v[0][0]}: {v[0][1]})")

