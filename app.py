"""Root entrypoint wrapper for the UIDAI Intelligence Studio dashboard."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for directory in [ROOT, ROOT / "src", ROOT / "dashboard"]:
    path_str = str(directory)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
