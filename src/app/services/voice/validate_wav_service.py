import hashlib
import io
import struct
import wave

from app.exceptions.invalid_input_error import InvalidInputError


class ValidateWavService:
    MAX_BYTES = 20 * 1024 * 1024

    def exec(self, content: bytes) -> dict:
        if not content or len(content) > self.MAX_BYTES:
            raise InvalidInputError()
        try:
            with wave.open(io.BytesIO(content), "rb") as audio:
                rate = audio.getframerate()
                frames = audio.getnframes()
                if (
                    audio.getnchannels() != 1
                    or audio.getsampwidth() != 2
                    or audio.getcomptype() != "NONE"
                    or rate not in (16000, 22050, 24000, 44100, 48000)
                    or not 3 * rate <= frames <= 180 * rate
                ):
                    raise InvalidInputError()
                pcm = audio.readframes(frames)
                if len(pcm) != frames * 2 or not any(pcm):
                    raise InvalidInputError()
            return dict(
                size=len(content),
                duration=round(frames * 1000 / rate),
                rate=rate,
                checksum=hashlib.sha256(content).hexdigest(),
            )
        except (wave.Error, EOFError, ValueError, struct.error) as error:
            raise InvalidInputError() from error
