import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from app.entities.dubbing_job.execution_result_entity import ExecutionResultEntity
from app.entities.dubbing_job.segment_entity import SegmentEntity
from app.exceptions.execution_timeout_error import ExecutionTimeoutError
from app.exceptions.lease_lost_error import LeaseLostError
from app.services.processing.private_result_storage import PrivateResultStorage
from app.services.voice.private_media_storage import PrivateMediaStorage


class SubprocessAudioExecution:
    def __init__(self, root, settings, health_callback=lambda: None, stopping=lambda: False):
        self.root = Path(root)
        self.settings = settings
        self.storage = PrivateMediaStorage(root)
        self.results = PrivateResultStorage(root)
        self.health_callback = health_callback
        self.stopping = stopping

    def supervise(self, arguments, renew):
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "PATH",
                "HOME",
                "PYTHONPATH",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONUNBUFFERED",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            }
        }
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        process = subprocess.Popen(
            [sys.executable, "-m", "app.engines.run_attempt", "--model-root", self.settings.model_root] + arguments,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        start = last = time.monotonic()
        try:
            while process.poll() is None:
                now = time.monotonic()
                self.health_callback()
                if self.stopping():
                    raise LeaseLostError()
                if now - start > self.settings.timeout_seconds:
                    raise ExecutionTimeoutError()
                if now - last >= self.settings.heartbeat_seconds:
                    if not renew():
                        raise LeaseLostError()
                    last = now
                time.sleep(0.25)
            if process.returncode:
                raise RuntimeError("Audio engine failed")
            if not renew():
                raise LeaseLostError()
        finally:
            # Also terminates ffmpeg descendants when the Python child has already exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def preflight(self):
        self.supervise(["--preflight"], lambda: True)

    def run(self, context, renew):
        attempts = self.root / ".attempts"
        attempts.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=attempts, prefix="job-") as directory:
            request = asdict(context)
            request["input_path"] = self.storage.path(context.input_key) if context.input_key else None
            request["sample_path"] = self.storage.path(context.sample_key) if context.sample_key else None
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request))
            path.chmod(0o600)
            self.supervise(["--request", str(path)], renew)
            manifest = Path(directory) / "result.json"
            if manifest.is_symlink() or manifest.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("Invalid manifest")
            result = json.loads(manifest.read_text())
            expected = (
                {"transcript.txt"}
                if context.operation == "transcribe"
                else ({"audio.wav"} if context.operation == "text_to_speech" else {"audio.wav", "transcript.txt"})
            )
            if set(result["files"]) != expected or len(result["files"]) != len(expected):
                raise ValueError("Missing outputs")
            segments = tuple(SegmentEntity(**item) for item in result["segments"])
            if any(s.start_ms < 0 or s.end_ms < s.start_ms for s in segments):
                raise ValueError("Invalid segments")
            files = tuple(self.results.save(context.office_id, directory, name) for name in result["files"])
            return ExecutionResultEntity(
                files, result["text"], result["language"], segments, result["engine"], result["model_version"]
            )
