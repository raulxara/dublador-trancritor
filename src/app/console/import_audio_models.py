"""Explicit import of existing local checkpoints; never downloads or accepts terms."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from app.config.worker_settings import WorkerSettings

FILES = {
    "xtts_v2": ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth", "tos_agreed.txt"),
    "whisper": ("tiny.pt",),
}


def main():
    destination = Path(WorkerSettings().model_root)
    if (destination / "manifest.json").exists():
        raise SystemExit("Cache already initialized; use a new volume for model replacement.")
    sources = Path("/import")
    for folder, names in FILES.items():
        for name in names:
            if not (sources / folder / name).is_file():
                raise SystemExit("Missing required model file: " + folder + "/" + name)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        staging = Path(temporary)
        manifest = {}
        for folder, names in FILES.items():
            (staging / folder).mkdir()
            for name in names:
                target = staging / folder / name
                shutil.copyfile(sources / folder / name, target)
                target.chmod(0o600)
                with target.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                manifest[folder + "/" + name] = digest
        for folder in FILES:
            if (destination / folder).exists():
                raise SystemExit("Partial cache found; use a new model volume.")
        for folder in FILES:
            (staging / folder).replace(destination / folder)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "manifest.json").replace(destination / "manifest.json")
    print("Model cache imported with SHA-256 manifest.")


if __name__ == "__main__":
    main()
