"""Child process entry point. Receives paths and model input, never DB credentials."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--request")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    from app.engines.local_audio_engine import LocalAudioEngine

    engine = LocalAudioEngine(args.model_root)
    if args.preflight:
        engine.preflight()
        return
    path = Path(args.request)
    result = engine.run(json.loads(path.read_text()), path.parent)
    (path.parent / "result.json").write_text(json.dumps(result), encoding="utf8")


if __name__ == "__main__":
    main()
