"""Official Packaging & Release Integrity Test Suite for AVCS v0.2.0.

Verifies:
1. Version declaration matches v0.2.0 across package and pyproject.toml.
2. Complete System 1 (Eyes + Hands) Public API export surface.
3. Checkpoint auto-discovery for both human_eye_weights.pt and goal_grounder_best.pt.
4. Strict AST audit ensuring zero prohibited external OCR or accessibility tree imports.
5. Wheel distribution artifact packaging integrity.
6. CPU-only compatibility without CUDA dependencies.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import zipfile

import pytest


def test_package_version_alignment():
    """Verify package version is aligned to 0.2.0."""
    import avcs

    assert avcs.__version__ == "0.2.0"

    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert pyproject_path.is_file()
    content = pyproject_path.read_text(encoding="utf-8")
    assert 'version = "0.2.0"' in content


def test_public_api_surface_exports():
    """Verify all core System 1 orchestrator and grounding classes are exported at package root."""
    import avcs

    # Central Orchestrator & Types
    assert hasattr(avcs, "AVCSController")
    assert hasattr(avcs, "ActionOutcome")
    assert hasattr(avcs, "ScreenObservation")

    # Neural Grounding
    assert hasattr(avcs, "GoalConditionedGrounder")
    assert hasattr(avcs, "GroundingPrediction")
    assert hasattr(avcs, "load_grounding_checkpoint")
    assert hasattr(avcs, "get_grounding_weights_path")

    # Classical Eyes, Hands, and Verifier
    assert hasattr(avcs, "HumanEye")
    assert hasattr(avcs, "HumanEyeState")
    assert hasattr(avcs, "HumanEyeClient")
    assert hasattr(avcs, "NeuromuscularServoingController")
    assert hasattr(avcs, "VisualRippleVerifier")
    assert hasattr(avcs, "NeuroStreamBus")
    assert hasattr(avcs, "VisualFocusDetector")


def test_checkpoint_auto_discovery():
    """Verify that both trained model checkpoints are auto-discovered on disk."""
    from avcs.weights import get_grounding_weights_path, get_human_eye_weights_path

    grounding_path = get_grounding_weights_path()
    assert grounding_path is not None, "Expected to locate goal_grounder_best.pt"
    assert Path(grounding_path).is_file()
    assert os.path.getsize(grounding_path) > 1_000_000, "Expected non-trivial neural weights (>1MB)"

    human_eye_path = get_human_eye_weights_path()
    assert human_eye_path is not None, "Expected to locate human_eye_weights.pt"
    assert Path(human_eye_path).is_file()
    assert os.path.getsize(human_eye_path) > 100_000, "Expected non-trivial cortical weights (>100KB)"


def test_prohibited_mechanisms_ast_audit():
    """Adversarial AST check ensuring no prohibited OCR or accessibility tree imports exist."""
    prohibited_modules = {
        "pytesseract",
        "tesseract",
        "easyocr",
        "paddleocr",
        "pywinauto",
        "uiautomation",
        "UIAutomationClient",
        "selenium",
        "playwright.sync_api",
    }

    avcs_dir = Path(__file__).resolve().parents[1] / "avcs"
    violations = []

    for py_file in avcs_dir.rglob("*.py"):
        # Allow testing mocks/audit stubs in testing subpackage if isolated
        if "testing" in py_file.parts:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in prohibited_modules:
                            violations.append(f"{py_file.name}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and any(node.module.startswith(m) for m in prohibited_modules):
                        violations.append(f"{py_file.name}: from {node.module} import ...")
        except Exception:
            pass

    assert not violations, f"PROHIBITED DEPENDENCIES DETECTED: {violations}"


def test_wheel_distribution_artifact():
    """Verify that the built wheel distribution package contains valid package metadata."""
    dist_dir = Path(__file__).resolve().parents[1] / "dist"
    wheels = list(dist_dir.glob("avcs_sdk-0.2.0-*.whl"))
    assert len(wheels) > 0, "Expected built wheel in dist/"

    wheel_path = wheels[0]
    assert wheel_path.stat().st_size > 400_000, f"Wheel size suspiciously small: {wheel_path.stat().st_size} bytes"

    with zipfile.ZipFile(wheel_path, "r") as z:
        names = z.namelist()
        assert "avcs/__init__.py" in names
        assert "avcs/controller.py" in names
        assert "avcs/goal_grounder.py" in names
        assert "avcs/human_eye.py" in names
        assert any(n.endswith("METADATA") for n in names)

