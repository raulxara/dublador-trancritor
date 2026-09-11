from pathlib import Path
from threading import Lock
from typing import Dict, Any, List

import whisper
import librosa

from app.config import ASR_MODEL_SIZE  # usamos "tiny" para validar

class ASREngine:
    """
    ASR via OpenAI Whisper (PyTorch puro, CPU). Estável em macOS Intel.
    """
    _instance = None
    _lock = Lock()

    def __init__(self):
        print(f"[ASR-OAI] loading whisper model={ASR_MODEL_SIZE} (CPU)...")
        # device="cpu" e fp16=False -> evita uso de half precision ausente em CPU
        self.model = whisper.load_model(ASR_MODEL_SIZE, device="cpu")
        print("[ASR-OAI] Model ready.")

    @classmethod
    def instance(cls) -> "ASREngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = ASREngine()
            return cls._instance

    def transcribe(self, audio_path: Path) -> Dict[str, Any]:
        print(f"[ASR-OAI] Transcribe start: {audio_path}")
        # Whisper faz resample internamente; nosso pipeline já entrega 16 kHz mono
        result = self.model.transcribe(
            str(audio_path),
            fp16=False,
            temperature=0,
            verbose=False
        )
        text = (result.get("text") or "").strip()
        lang = result.get("language")
        # duração só para referência
        try:
            y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            duration = float(len(y) / sr)
        except Exception:
            duration = 0.0

        segs: List[Dict[str, Any]] = []
        for s in result.get("segments", []):
            segs.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": (s.get("text") or "").strip()
            })
        print(f"[ASR-OAI] Transcribe done. TextLen={len(text)}")
        return {"language": lang, "duration": duration, "segments": segs, "text": text}
