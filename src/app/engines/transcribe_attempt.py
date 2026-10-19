"""Private ASR subprocess. Its memory is released before voice synthesis starts."""
import json
import sys
from pathlib import Path


def main():
    import whisper
    if sys.argv[1] == "--preflight":
        whisper.load_model(str(Path(sys.argv[2]) / "whisper" / "tiny.pt"), device="cpu")
        return
    request_path, output_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text())
    model = whisper.load_model(str(Path(request["model_root"]) / "whisper" / "tiny.pt"), device="cpu")
    result = model.transcribe(request["input_path"], language=request["language"], fp16=False, temperature=0, verbose=False)
    output_path.write_text(json.dumps(result), encoding="utf8")
    output_path.chmod(0o600)


if __name__ == "__main__":
    main()
