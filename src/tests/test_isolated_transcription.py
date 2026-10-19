import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from app.engines.isolated_transcription import transcribe_isolated


def test_transcription_uses_separate_process_and_removes_private_files(tmp_path):
    def run(args, **kwargs):
        request = json.loads(Path(args[-2]).read_text())
        assert request["language"] == "pt"
        assert kwargs["check"] is True
        Path(args[-1]).write_text(json.dumps({"text": "teste", "language": "pt", "segments": []}))
    with patch("subprocess.run", side_effect=run):
        result = transcribe_isolated("/models", "/private/input.wav", "pt", tmp_path)
    assert result["text"] == "teste"
    assert list(tmp_path.iterdir()) == []


def test_failed_transcription_removes_private_request(tmp_path):
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "asr")):
        with pytest.raises(subprocess.CalledProcessError):
            transcribe_isolated("/models", "/private/input.wav", None, tmp_path)
    assert list(tmp_path.iterdir()) == []
