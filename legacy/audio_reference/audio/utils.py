import subprocess, json, math
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import soundfile as sf
import librosa

from app.config import SAMPLE_RATE
from pathlib import Path
import soundfile as sf

def get_audio_duration_sec(path: Path) -> float:
    """Retorna a duração do arquivo de áudio em segundos (float)."""
    info = sf.info(str(path))
    if info.samplerate and info.frames:
        return float(info.frames) / float(info.samplerate)
    return 0.0

def run(cmd: list) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, text=True, capture_output=True)
    return p.returncode, p.stdout, p.stderr

def sniff_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".mp4", ".mov"]:
        return ext[1:]
    return "unknown"

def ensure_wav_mono_22050(src: Path, dst: Path) -> Path:
    """Converte para WAV mono 22.05 kHz (padrão do TTS)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-sample_fmt", "s16",
        str(dst)
    ]
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg falhou ao converter '{src}': {err}")
    return dst

def ensure_wav_mono_sr(src: Path, dst: Path, sr: int) -> Path:
    """Converte qualquer mídia para WAV mono, SR definido (ex.: 16000 p/ ASR)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1", "-ar", str(sr),
        "-sample_fmt", "s16",
        str(dst)
    ]
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg falhou ao converter '{src}': {err}")
    return dst

def ensure_wav_mono_16000(src: Path, dst: Path) -> Path:
    """Atalho para ASR: WAV mono 16 kHz."""
    return ensure_wav_mono_sr(src, dst, sr=16000)

def trim_audio(src: Path, dst: Path, start_sec: float = 0.0, duration_sec: float = 30.0) -> Path:
    """Corta um trecho do áudio com ffmpeg (por padrão, primeiros 30s)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # tentativa rápida (copy)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec}",
        "-t", f"{duration_sec}",
        "-i", str(src),
        "-c", "copy",
        str(dst)
    ]
    code, out, err = run(cmd)
    if code != 0:
        # fallback: força reencode para evitar problemas de copy
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec}",
            "-t", f"{duration_sec}",
            "-i", str(src),
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            str(dst)
        ]
        code2, out2, err2 = run(cmd)
        if code2 != 0:
            raise RuntimeError(f"ffmpeg falhou ao cortar '{src}': {err2}")
    return dst

def measure_audio_stats(wav_path: Path) -> Dict:
    """Mede duração, RMS, pico, clipping aprox, % silêncio e LUFS estimado."""
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    dur = librosa.get_duration(y=y, sr=sr)
    rms = float(np.sqrt(np.mean(y**2))) if len(y) else 0.0
    peak = float(np.max(np.abs(y))) if len(y) else 0.0

    clip_ratio = float(np.mean(np.abs(y) > 0.999)) if len(y) else 0.0

    frame_len = int(0.02 * sr) or 1  # 20 ms
    if len(y) >= frame_len:
        frames = np.array([np.sqrt(np.mean(y[i:i+frame_len]**2))
                           for i in range(0, len(y), frame_len)])
        silence_ratio = float(np.mean(frames < 0.001))
    else:
        silence_ratio = 1.0

    lufs_est = -0.691 + 10 * math.log10(rms**2 + 1e-12)  # aprox

    return {
        "duration_sec": round(dur, 3),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "clip_ratio": round(clip_ratio, 6),
        "silence_ratio": round(silence_ratio, 6),
        "lufs_est": round(lufs_est, 2),
        "sr": sr,
    }
