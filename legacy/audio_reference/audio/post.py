# app/audio/post.py
from __future__ import annotations
from pathlib import Path
import math
import shutil
import subprocess

from app.config import MP3_BITRATE  # usa o bitrate configurado na tua app


def _run_ffmpeg(args: list[str]) -> None:
    """Executa ffmpeg e lança uma exceção com stderr se falhar."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg não encontrado. Instale com 'brew install ffmpeg' (macOS) "
            "ou adicione ao PATH."
        ) from e

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falhou:\n{proc.stderr}")


def _decompose_atempo_factor(f: float) -> list[float]:
    """
    Decompõe um fator qualquer em uma cadeia de valores dentro de [0.5, 2.0]
    para o filtro atempo do ffmpeg.
    """
    if f <= 0:
        return [1.0]
    chain = []
    remaining = f
    while remaining < 0.5:
        chain.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        chain.append(2.0)
        remaining /= 2.0
    if not math.isclose(remaining, 1.0, rel_tol=1e-3):
        chain.append(remaining)
    if not chain:
        chain = [1.0]
    return chain


def apply_speed_pitch(
    in_wav: Path,
    out_wav: Path,
    *,
    speed: float = 1.0,
    semitones: int = 0,
) -> None:
    """
    Ajusta velocidade (tempo) e afinação (pitch) com **alta qualidade** via FFmpeg.
    - Pitch: usa truque asetrate+aresample+atempo para preservar duração.
    - Speed: usa cadeia de 'atempo' (0.5..2.0) para valores fora desse intervalo.
    - Saída: mono, 44.1 kHz, 16-bit PCM (compatível com teu fluxo atual).
    """
    in_wav = Path(in_wav)
    out_wav = Path(out_wav)

    if not in_wav.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {in_wav}")

    # Se nada para fazer, apenas copia como 44.1 kHz / PCM16
    if (math.isclose(speed, 1.0, rel_tol=1e-6) or speed <= 0) and semitones == 0:
        _run_ffmpeg([
            "-hide_banner", "-loglevel", "error",
            "-i", str(in_wav),
            "-ac", "1", "-ar", "44100", "-sample_fmt", "s16",
            str(out_wav),
        ])
        return

    filters: list[str] = []

    # 1) Pitch (em semitons) preservando a duração
    if semitones != 0:
        pf = 2.0 ** (semitones / 12.0)  # fator de frequência
        # muda o "sample rate efetivo" para alterar pitch...
        # e volta a duração ao normal com atempo = 1/pf
        filters.append(f"asetrate=44100*{pf:.8f}")
        filters.append("aresample=44100")
        for f in _decompose_atempo_factor(1.0 / pf):
            filters.append(f"atempo={f:.8f}")

    # 2) Speed global (tempo)
    if not math.isclose(speed, 1.0, rel_tol=1e-6) and speed > 0:
        for f in _decompose_atempo_factor(speed):
            filters.append(f"atempo={f:.8f}")

    filter_arg = ",".join(filters) if filters else "anull"

    _run_ffmpeg([
        "-hide_banner", "-loglevel", "error",
        "-i", str(in_wav),
        "-filter:a", filter_arg,
        "-ac", "1", "-ar", "44100", "-sample_fmt", "s16",
        str(out_wav),
    ])


def wav_to_mp3(in_wav: Path, out_mp3: Path, bitrate: str | None = None) -> None:
    """
    Converte WAV para MP3 (libmp3lame) usando o bitrate configurado.
    """
    br = bitrate or MP3_BITRATE
    _run_ffmpeg([
        "-hide_banner", "-loglevel", "error",
        "-i", str(in_wav),
        "-vn",
        "-c:a", "libmp3lame",
        "-b:a", str(br),
        str(out_mp3),
    ])
