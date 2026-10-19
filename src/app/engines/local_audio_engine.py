import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from app.engines.audio_filters import AudioFilters
from app.engines.text_chunks import synthesis_chunks
from app.engines.isolated_transcription import transcribe_isolated, preflight_transcription


class LocalAudioEngine:
    """Offline cached models. Speech conversion uses ASR followed by reference-conditioned synthesis."""

    def __init__(self, model_root):
        self.root = Path(model_root)
        self.xtts = None

    def load_xtts(self):
        if self.xtts is None:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            directory = self.root / "xtts_v2"
            config = XttsConfig()
            config.load_json(str(directory / "config.json"))
            self.xtts = Xtts.init_from_config(config)
            self.xtts.load_checkpoint(config, checkpoint_dir=str(directory), use_deepspeed=False)
            self.xtts.eval()
        return self.xtts

    def preflight(self):
        manifest = self.root / "manifest.json"
        if manifest.is_file():
            for name, digest in json.loads(manifest.read_text()).items():
                path = (self.root / name).resolve()
                if not path.is_relative_to(self.root.resolve()):
                    raise ValueError("Invalid model path")
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                        raise ValueError("Model checksum mismatch")
        preflight_transcription(self.root)
        self.load_xtts()

    def run(self, request, directory):
        directory = Path(directory)
        operation = request["operation"]
        language = request.get("language")
        if language:
            language = language.lower().replace("_", "-").split("-")[0]
        transcript = None
        segments = []
        if operation != "text_to_speech":
            result = transcribe_isolated(self.root, request["input_path"], language, directory)
            transcript = result["text"].strip()
            language = result["language"]
            if not transcript:
                raise ValueError("Empty transcription")
            segments = [
                dict(
                    start_ms=max(0, round(s["start"] * 1000)),
                    end_ms=max(0, round(s["end"] * 1000)),
                    text=s["text"].strip(),
                )
                for s in result["segments"]
                if s["text"].strip()
            ]
            (directory / "transcript.txt").write_text(transcript, encoding="utf8")
        files = ["transcript.txt"] if transcript is not None else []
        if operation != "transcribe":
            model = self.load_xtts()
            if language not in model.config.languages:
                raise ValueError("Unsupported language")
            latent, embedding = model.get_conditioning_latents(audio_path=[request["sample_path"]])

            def synthesize(text, index):
                raw = directory / f"raw-{index}.wav"
                cooked = directory / f"cooked-{index}.wav"
                # Write bounded phrases incrementally instead of decoding a long
                # utterance into large intermediate vocoder tensors at once.
                with sf.SoundFile(str(raw), mode="w", samplerate=24000, channels=1, subtype="PCM_16") as output:
                    for phrase in synthesis_chunks(text):
                        result = model.inference(phrase, language, latent, embedding, enable_text_splitting=True)
                        output.write(result["wav"])
                        del result
                gc.collect()
                AudioFilters.convert(raw, cooked, request["speed"], request["pitch_semitones"])
                return cooked

            if operation == "speech_to_speech" and request["preserve_timing"]:
                source_duration = sf.info(request["input_path"]).duration
                audio = np.zeros(round(source_duration * 24000), dtype=np.float32)
                for index, segment in enumerate(segments):
                    start = min(len(audio), round(segment["start_ms"] * 24))
                    end = min(len(audio), round(segment["end_ms"] * 24))
                    if end <= start:
                        continue
                    cooked = synthesize(segment["text"], index)
                    duration = (end - start) / 24000
                    fitted = directory / f"fitted-{index}.wav"
                    AudioFilters.convert(
                        cooked, fitted, speed=sf.info(str(cooked)).duration / duration, duration=duration
                    )
                    wave, _ = sf.read(str(fitted), dtype="float32")
                    audio[start : start + min(len(wave), end - start)] += wave[: end - start]
                sf.write(str(directory / "audio.wav"), np.clip(audio, -1, 1), 24000, subtype="PCM_16")
            else:
                cooked = synthesize(request["input_text"] if operation == "text_to_speech" else transcript, 0)
                cooked.replace(directory / "audio.wav")
            files.append("audio.wav")
        return dict(
            files=files,
            text=transcript,
            language=language,
            segments=segments,
            engine="whisper"
            if operation == "transcribe"
            else ("xtts" if operation == "text_to_speech" else "whisper+xtts"),
            model_version="tiny-20240930/xtts-v2",
        )
