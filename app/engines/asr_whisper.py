from pathlib import Path
from threading import Lock
from typing import Dict, List, Any

from faster_whisper import WhisperModel  # pip install faster-whisper
from app.config import ASR_MODEL_SIZE, ASR_COMPUTE_TYPE

class ASREngine:
    """
    Singleton para transcrição local com faster-whisper (CPU).
    - Usa menos threads para não 'engasgar' a UI/CPU.
    - Faz chunking curto para ficar responsivo.
    """
    _instance = None
    _lock = Lock()

    def __init__(self):
        # Limite de threads ajuda MUITO em máquinas Intel.
        cpu_threads = 2  # ajuste se quiser (2-4 geralmente é ótimo)
        print(f"[ASR] Loading faster-whisper model={ASR_MODEL_SIZE} compute_type={ASR_COMPUTE_TYPE} cpu_threads={cpu_threads} ...")
        self.model = WhisperModel(
            ASR_MODEL_SIZE,
            device="cpu",
            compute_type=ASR_COMPUTE_TYPE,
            cpu_threads=cpu_threads,
        )
        print("[ASR] Model ready.")

    @classmethod
    def instance(cls) -> "ASREngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = ASREngine()
            return cls._instance

    def transcribe(self, audio_path: Path) -> Dict[str, Any]:
        print(f"[ASR] Transcribe start: {audio_path}")
        # Parâmetros de transcrição para ficar rápido e estável
        segments, info = self.model.transcribe(
            str(audio_path),
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            beam_size=1,
            condition_on_previous_text=False,
            chunk_length=15,            # processa em janelas <=15s
            without_timestamps=True,    # não precisa timestamps agora
        )
        texts: List[str] = []
        for s in segments:
            if s.text:
                texts.append(s.text.strip())
        full_text = " ".join(texts).strip()
        print(f"[ASR] Transcribe done. Duration={info.duration:.2f}s, TextLen={len(full_text)}")
        return {
            "language": info.language,  # ex.: "pt"
            "duration": float(info.duration),
            "segments": [],             # omitimos detalhes pq without_timestamps=True
            "text": full_text,
        }
