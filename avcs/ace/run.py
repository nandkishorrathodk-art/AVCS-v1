"""Direct CLI launcher for the Ace-Control-Medium Autopilot."""
import os
import sys
from pathlib import Path

# Auto-detect repo .venv if running from system python where dependencies are missing
_repo_root = Path(__file__).resolve().parents[2]
_venv_python = _repo_root / ".venv" / "Scripts" / "python.exe"

if (
    os.environ.get("_AVCS_REEXECED") != "1"
    and _venv_python.is_file()
    and Path(sys.executable).resolve() != _venv_python.resolve()
):
    try:
        import PIL
        import torch
    except ImportError:
        import subprocess
        # Transparently re-exec using the repo's configured virtual environment
        env = dict(os.environ, _AVCS_REEXECED="1")
        sys.exit(subprocess.call([str(_venv_python), "-m", "avcs.ace.run", *sys.argv[1:]], env=env))

from avcs.ace.runner import main

if __name__ == "__main__":
    main()

