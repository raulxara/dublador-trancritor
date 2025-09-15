from pathlib import Path
from threading import Lock
import os
import re
import shutil
import numpy as np
import soundfile as sf

import torch
from TTS.api import TTS  # pip install TTS

from app.config import SAMPLE_RATE

# ---------------- Normalização: "." / "…" -> ";" + divisão em segmentos ----------------
# Usamos um marcador que o TTS não fala; depois dividimos o áudio nesses pontos.
_MARK = "[[PAUSE_AFTER_DOT]]"
_MARK_RE = re.compile(r"\s*" + re.escape(_MARK) + r"\s*")

def _normalize_points(text: str) -> str:
    """
    Converte apenas pontos finais em vírgula dupla + marcador:
      "frase."  -> "frase; [[PAUSE_AFTER_DOT]]"
      "frase…"  -> "frase; [[PAUSE_AFTER_DOT]]"
    NÃO altera decimais "2.5" nem pontos no meio de siglas (heurística simples).
    """
    if not text:
        return ""
    s = text

    # "…" final -> ";"
    s = re.sub(r"…(?=\s|$)", "; " + _MARK, s)

    # "." final (não após dígito) -> ";"
    s = re.sub(r"(?<!\d)\.(?=\s|$)", "; " + _MARK, s)

    return s

def _split_segments(text: str) -> list[str]:
    """
    1) Normaliza pontos para ";" + marcador;
    2) Corta SOMENTE nos marcadores (ou seja, onde havia ".").
    Mantém TODO o resto da pontuação (! ? : ; , aspas etc).
    """
    norm = _normalize_points(text)
    parts = [p.strip() for p in _MARK_RE.split(norm) if p and p.strip()]
    return parts

def _join_with_silence(chunks: list[np.ndarray], sr: int, pause_ms: int = 120) -> np.ndarray:
    """
    Junta os pedaços inserindo uma pausa curta entre eles.
    pause_ms padrão: 120 ms (curtinha, como solicitado).
    """
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    if pause_ms <= 0:
        return np.concatenate(chunks).astype(np.float32)
    gap = np.zeros(int(sr * (pause_ms / 1000.0)), dtype=np.float32)
    out = []
    for i, ch in enumerate(chunks):
        out.append(ch.astype(np.float32))
        if i < len(chunks) - 1:
            out.append(gap)
    return np.concatenate(out).astype(np.float32)

# ---------------- Engine ----------------

class XTTSEngine:
    """
    XTTS-v2 com síntese por segmentos:
    - Substitui SÓ o ponto final/reticências por ";" e divide nesses pontos;
    - Insere uma PAUSA CURTA entre segmentos;
    - Mantém todas as outras pontuações;
    - Tenta desativar split/normalização interna do Coqui para não recolocar ".".
    """

    _instance = None
    _lock = Lock()

    def __init__(self, device: str | None = None):
        forced = os.getenv("DUBBER_TORCH_DEVICE")
        if forced:
            device = forced
        if device is None:
            device = "cpu"

        self.device = device
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

        if self.device == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        self.tts = TTS(self.model_name).to(self.device)

    @classmethod
    def instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = XTTSEngine()
            return cls._instance

    # ====== Importante: redirecionar o modo simples para o "smart" ======
    def synthesize_to_file(self, text: str, speaker_wav: Path, language: str, out_path: Path) -> Path:
        """
        Redirecionado para o modo SMART para garantir a troca de ponto->";" + pausa.
        (Isso cobre quaisquer chamadas antigas do app que ainda usem este método.)
        """
        return self.synthesize_smart_to_file(text, speaker_wav, language, out_path, pause_ms=120)

    # ====== Interno: chamar TTS tentando desativar splits ======
    def _tts_to_file_nosplit(self, text: str, file_path: Path, speaker_wav: Path, language: str):
        """
        Chama o TTS tentando desativar splits/normalização interna.
        """
        safe_text = (text or "").strip() + " "  # espaço final ajuda no EOS
        try:
            self.tts.tts_to_file(
                text=safe_text,
                file_path=str(file_path),
                speaker_wav=str(speaker_wav),
                language=language,
                split_sentences=False,
                enable_text_splitting=False,
            )
        except TypeError:
            # versões que não aceitam os kwargs acima
            self.tts.tts_to_file(
                text=safe_text,
                file_path=str(file_path),
                speaker_wav=str(speaker_wav),
                language=language
            )

    def synthesize_smart_to_file(
        self,
        text: str,
        speaker_wav: Path,
        language: str,
        out_path: Path,
        pause_ms: int = 120
    ) -> Path:
        """
        Pipeline:
        - Converte "."/ "…" finais para ";" + marcador;
        - Divide SOMENTE nesses pontos;
        - Sintetiza cada parte sem splits internos;
        - Junta com pausa curta entre as partes.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)

        segments: list[str] = _split_segments(text)
        if not segments:
            sf.write(str(out_path), np.zeros(1, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")
            return out_path

        tmp_dir = out_path.parent / "_tmp_xtts"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        chunks: list[np.ndarray] = []
        sr_seen = None
        try:
            # debug: o que vai para o TTS (já com ";" no lugar dos ".")
            print(f"[TTS-SMART] {len(segments)} segmentos:", segments)

            for i, seg_text in enumerate(segments):
                seg_file = tmp_dir / f"seg_{i:03d}.wav"
                self._tts_to_file_nosplit(seg_text, seg_file, speaker_wav, language)

                wav, sr = sf.read(str(seg_file), dtype="float32", always_2d=False)
                if sr_seen is None:
                    sr_seen = sr
                elif sr != sr_seen:
                    raise RuntimeError(f"SR inconsistente: {sr} vs {sr_seen}")
                if isinstance(wav, np.ndarray) and wav.ndim > 1:
                    wav = wav[:, 0]
                chunks.append(wav.astype(np.float32))

            joined = _join_with_silence(chunks, sr_seen or SAMPLE_RATE, pause_ms=pause_ms)

            # normalização leve
            if joined.size > 0:
                peak = float(np.max(np.abs(joined)))
                if peak > 0.99:
                    joined = 0.99 * joined / peak

            sf.write(str(out_path), joined, sr_seen or SAMPLE_RATE, subtype="PCM_16")
            return out_path
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
