import json
import subprocess
import sys
import tempfile
from pathlib import Path


def transcribe_isolated(model_root, input_path, language, directory):
    with tempfile.TemporaryDirectory(dir=directory, prefix="asr-") as temporary:
        request = Path(temporary) / "input.json"
        result = Path(temporary) / "result.json"
        request.write_text(json.dumps(dict(model_root=str(model_root), input_path=input_path, language=language)))
        request.chmod(0o600)
        subprocess.run([sys.executable, "-m", "app.engines.transcribe_attempt", str(request), str(result)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return json.loads(result.read_text())


def preflight_transcription(model_root):
    subprocess.run([sys.executable, "-m", "app.engines.transcribe_attempt", "--preflight", str(model_root)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
