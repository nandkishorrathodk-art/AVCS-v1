"""Direct CLI launcher for the Ace-Control-Medium Autopilot."""
import sys
from pathlib import Path

# Auto-detect repo .venv if running from system python where dependencies are missing
_repo_root = Path(__file__).resolve().parents[2]
_venv_python = _repo_root / ".venv" / "Scripts" / "python.exe"

if _venv_python.is_file() and Path(sys.executable).resolve() != _venv_python.resolve():
    try:
        import PIL
        import torch
    except ImportError:
        import subprocess
        # Transparently re-exec using the repo's configured virtual environment
        sys.exit(subprocess.call([str(_venv_python), "-m", "avcs.ace.run", *sys.argv[1:]]))

from avcs.ace.runner import main

if __name__ == "__main__":
    main()

