import os
import wave
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from app.entities.dubbing_job.result_file_entity import ResultFileEntity


class PrivateResultStorage:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def save(self, office_id, directory, name):
        if name not in ("audio.wav", "transcript.txt"):
            raise ValueError("Invalid artifact")
        path = Path(directory) / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("Invalid artifact")
        content = path.read_bytes()
        if not content:
            raise ValueError("Empty artifact")
        rate = duration = channels = None
        if name.endswith(".wav"):
            with wave.open(str(path), "rb") as audio:
                rate = audio.getframerate()
                channels = audio.getnchannels()
                frames = audio.getnframes()
                data = audio.readframes(frames)
                if (
                    rate != 24000
                    or channels != 1
                    or audio.getsampwidth() != 2
                    or frames < 1
                    or len(data) != frames * 2
                    or not any(data)
                ):
                    raise ValueError("Invalid generated WAV")
                duration = round(frames * 1000 / rate)
        else:
            content.decode("utf8")
        identifier = str(uuid4())
        extension = name.split(".")[-1]
        key = sha256(office_id.encode()).hexdigest() + "/" + identifier + "." + extension
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Exclusive creation; a result UUID is never reused by another attempt.
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        return ResultFileEntity(
            identifier,
            key,
            extension,
            "audio" if extension == "wav" else "transcript",
            len(content),
            sha256(content).hexdigest(),
            duration,
            rate,
            channels,
        )
