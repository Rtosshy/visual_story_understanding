from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent

if not (PROJECT_ROOT / "pyproject.toml").exists():
    raise RuntimeError("Project root not found.")

DATA_ROOT = PROJECT_ROOT / "dataset"

VIST_ROOT = DATA_ROOT / "vist"
VIST_IMAGE_ROOT = VIST_ROOT / "image"
VIST_JSON_ROOT = VIST_ROOT / "json"

ORIGINAL_ROOT = DATA_ROOT / "original"


OUTPUT_ROOT = PROJECT_ROOT / "output"

DOTENV_PATH = PROJECT_ROOT / ".env"
