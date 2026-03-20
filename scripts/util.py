import sys
from pathlib import Path

PARENT_DIR = Path(sys.argv[0]).parent
PROJECT_DIR = Path('..' if PARENT_DIR == Path('.') else PARENT_DIR / '..').resolve()
