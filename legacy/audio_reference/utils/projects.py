from datetime import datetime
from pathlib import Path
from app.config import PROJECTS_DIR

def new_job_dir(prefix: str = "tts") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = PROJECTS_DIR / f"{prefix}-{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d
