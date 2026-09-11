import json
import subprocess
import tempfile
from pathlib import Path
from threading import BoundedSemaphore

from app.exceptions.conflict_error import ConflictError
from app.exceptions.invalid_input_error import InvalidInputError
from app.services.voice.validate_wav_service import ValidateWavService


class NormalizeMediaService:
    """Decode local uploaded bytes with bounded time and audio duration."""

    slots = BoundedSemaphore(2)
    formats = {"audio/mpeg": "mp3", "video/mp4": "mov"}

    def exec(self, content: bytes, content_type: str) -> bytes:
        if len(content) > ValidateWavService.MAX_BYTES:
            raise InvalidInputError()
        if content_type in ("audio/wav", "audio/x-wav"):
            ValidateWavService().exec(content)
            return content
        if content_type not in self.formats:
            raise InvalidInputError()
        if not self.slots.acquire(blocking=False):
            raise ConflictError()
        try:
            with tempfile.TemporaryDirectory(prefix="dubber-decode-") as temporary:
                source = Path(temporary) / "input"
                target = Path(temporary) / "normalized.wav"
                source.write_bytes(content)
                source.chmod(0o600)
                demuxer = self.formats[content_type]
                options = ["-protocol_whitelist", "file,pipe", "-format_whitelist", demuxer, "-f", demuxer]
                probe = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        *options,
                        "-show_entries",
                        "format=duration:stream=codec_type",
                        "-of",
                        "json",
                        str(source),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
                details = json.loads(probe.stdout)
                if not any(stream.get("codec_type") == "audio" for stream in details.get("streams", [])):
                    raise InvalidInputError()
                duration = details.get("format", {}).get("duration")
                if duration is not None and float(duration) > 180:
                    raise InvalidInputError()
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-y",
                        "-threads",
                        "2",
                        *options,
                        "-i",
                        str(source),
                        "-map",
                        "0:a:0",
                        "-vn",
                        "-sn",
                        "-dn",
                        "-t",
                        "181",
                        "-filter_threads",
                        "2",
                        "-threads",
                        "2",
                        "-ac",
                        "1",
                        "-ar",
                        "24000",
                        "-c:a",
                        "pcm_s16le",
                        str(target),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60,
                )
                if target.stat().st_size > ValidateWavService.MAX_BYTES:
                    raise InvalidInputError()
                normalized = target.read_bytes()
                ValidateWavService().exec(normalized)
                return normalized
        except (subprocess.SubprocessError, OSError, ValueError, KeyError):
            raise InvalidInputError() from None
        finally:
            self.slots.release()
