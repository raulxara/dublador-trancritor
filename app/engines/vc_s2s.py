# app/engines/vc_s2s.py
from __future__ import annotations
from pathlib import Path
from threading import Lock
import subprocess
import tempfile
import shutil
import math

import numpy as np
import soundfile as sf

from app.audio.utils import ensure_wav_mono_16000
from app.audio.post import wav_to_mp3  # pode ser útil externamente
from app.config import SAMPLE_RATE_TTS, SAMPLE_RATE, DATA_ROOT
from app.engines.tts_xtts import XTTSEngine

# (Opcional futuro) placeholder para backend OpenVoice
_HAS_OPENVOICE = False
try:
    # se você vendorizar o OpenVoice depois, mude este import
    import openvoice  # noqa: F401
    _HAS_OPENVOICE = True
except Exception:
    _HAS_OPENVOICE = False


def _ffmpeg_atempo(in_wav: Path, out_wav: Path, speed_factor: float) -> None:
    """
    Ajusta o tempo (duração) preservando o pitch via ffmpeg atempo.
    speed_factor: >1 acelera (encurta), <1 desacelera (alongar).
    """
    # ffmpeg atempo aceita 0.5..2.0 – encadeamos se necessário
    f = float(max(0.1, min(10.0, speed_factor)))
    filters = []
    while f > 2.0:
        filters.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        filters.append("atempo=0.5")
        f *= 2.0
    filters.append(f"atempo={f:.6f}")
    flt = ",".join(filters)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(in_wav),
        "-filter:a", flt,
        str(out_wav)
    ]
    subprocess.run(cmd, check=True)


def _read_duration(wav_path: Path) -> float:
    with sf.SoundFile(str(wav_path), "r") as f:
        return f.frames / float(f.samplerate)


def _resample_to(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data.astype(np.float32, copy=False)
    try:
        import librosa
        y = librosa.resample(data.astype(np.float32), orig_sr=sr_in, target_sr=sr_out, res_type="soxr_hq")
        return y.astype(np.float32)
    except Exception:
        # fallback linear simples
        import numpy as np
        duration = len(data) / sr_in
        n_out = int(round(duration * sr_out))
        return np.interp(np.linspace(0, len(data)-1, n_out), np.arange(len(data)), data).astype(np.float32)


class VCEngine:
    """
    S2S (speech-to-speech) engine com dois backends:
      - 'openvoice' (opcional, zero-shot VC real)  -> TODO vendor
      - 'prosody'   (padrão) usa ASR+TTS+atempo por trecho para casar tempos/pausas.

    API:
      convert(src_audio, speaker_wav, out_wav, keep_sr=True, normalize=True) -> Path
    """
    _instance = None
    _lock = Lock()

    def __init__(self, backend: str | None = None):
        if backend is None:
            backend = "openvoice" if _HAS_OPENVOICE else "prosody"
        self.backend = backend

    @classmethod
    def instance(cls, backend: str | None = None) -> "VCEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = VCEngine(backend)
            return cls._instance

    # -------------- API pública --------------
    def convert(self,
                src_audio: Path,
                speaker_wav: Path,
                out_wav: Path,
                language: str = "pt",
                keep_sr: bool = True,
                normalize: bool = True) -> Path:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "openvoice" and _HAS_OPENVOICE:
            # placeholder – deixamos hookado para quando vendorizar o OpenVoice
            return self._convert_openvoice_placeholder(src_audio, speaker_wav, out_wav)
        else:
            return self._convert_prosody_match(src_audio, speaker_wav, out_wav, language, keep_sr, normalize)

    # -------------- Backend B (prosódia forçada) --------------
    def _convert_prosody_match(self,
                               src_audio: Path,
                               speaker_wav: Path,
                               out_wav: Path,
                               language: str,
                               keep_sr: bool,
                               normalize: bool) -> Path:
        """
        1) ASR com timestamps (faster-whisper) -> segmentos (start,end,text)
        2) TTS XTTS por segmento (com sua voz-base) -> seg_tts.wav
        3) Ajuste com ffmpeg atempo para cada segmento caber no intervalo original
        4) Inserir silenços medidos entre segmentos
        5) Concatenar tudo e ressamplar (opcional) para SR original
        """
        # a) preparar ASR: 16 kHz mono
        tmp_dir = Path(tempfile.mkdtemp(prefix="vc_s2s_"))
        try:
            src_16k = tmp_dir / "src16k.wav"
            ensure_wav_mono_16000(src_audio, src_16k)

            # b) rodar ASR com timestamps
            try:
                from faster_whisper import WhisperModel
            except Exception as e:
                raise RuntimeError("Instale 'faster-whisper' para S2S (pip install faster-whisper).") from e

            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, _info = model.transcribe(str(src_16k), task="transcribe", vad_filter=True)
            segs = []
            for seg in segments:
                txt = (seg.text or "").strip()
                start = float(seg.start)
                end = float(seg.end)
                if end > start and txt:
                    segs.append((start, end, txt))

            if not segs:
                raise RuntimeError("ASR não retornou segmentos com texto. Tente um áudio mais limpo.")

            # c) gerar TTS por segmento
            xtts = XTTSEngine.instance()
            seg_files_fit: list[Path] = []

            # SR de saída alvo
            if keep_sr:
                # ler SR do original
                with sf.SoundFile(str(src_audio), "r") as f:
                    sr_out = int(f.samplerate)
            else:
                sr_out = SAMPLE_RATE  # 22050 (XTTS)

            for i, (start, end, txt) in enumerate(segs):
                # limpeza leve para evitar falar "ponto"
                # usamos o caminho "inteligente" do XTTS que já trata pontuação final
                seg_raw = tmp_dir / f"seg_raw_{i:03d}.wav"
                seg_fit = tmp_dir / f"seg_fit_{i:03d}.wav"

                # gerar com XTTS (usa sua voz-base) – pausa interna já é curta
                xtts.synthesize_smart_to_file(
                    text=txt,
                    speaker_wav=speaker_wav,
                    language=language,
                    out_path=seg_raw,
                    pause_ms=120
                )

                # d) forçar duração do segmento para caber em (end-start)
                tts_dur = _read_duration(seg_raw)
                target = max(0.06, end - start)  # não deixar alvo < 60 ms
                factor = max(0.25, min(4.0, tts_dur / target))  # atempo factor

                if abs(factor - 1.0) < 0.03:
                    # quase igual: só copiar
                    shutil.copyfile(seg_raw, seg_fit)
                else:
                    _ffmpeg_atempo(seg_raw, seg_fit, speed_factor=factor)

                # garantir SR unificado
                wav, sr = sf.read(str(seg_fit), dtype="float32")
                if wav.ndim > 1:
                    wav = wav[:, 0]
                wav = _resample_to(wav, sr, sr_out)
                # normalizar leve por segurança (evita clipping acumulado)
                peak = float(np.max(np.abs(wav))) if wav.size else 0.0
                if normalize and peak > 0.99:
                    wav = 0.99 * wav / peak
                # escrever de volta padronizado
                sf.write(str(seg_fit), wav.astype(np.float32), sr_out, subtype="PCM_16")
                seg_files_fit.append(seg_fit)

            # e) concatenar com silenços medidos
            # gaps: (start_next - end_current)
            out_wave = []
            for i, seg_path in enumerate(seg_files_fit):
                wav, sr = sf.read(str(seg_path), dtype="float32")
                if wav.ndim > 1:
                    wav = wav[:, 0]
                out_wave.append(wav.astype(np.float32))

                if i < len(segs) - 1:
                    gap = max(0.0, segs[i+1][0] - segs[i][1])
                    if gap > 1e-3:
                        gap_len = int(round(gap * sr_out))
                        out_wave.append(np.zeros(gap_len, dtype=np.float32))

            out_arr = np.concatenate(out_wave) if out_wave else np.zeros(1, dtype=np.float32)
            # normalização de saída (-1 dBFS aprox)
            if normalize and out_arr.size:
                peak = float(np.max(np.abs(out_arr)))
                if peak > 0.99:
                    out_arr = 0.99 * out_arr / peak

            sf.write(str(out_wav), out_arr.astype(np.float32), sr_out, subtype="PCM_16")
            return out_wav

        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    # -------------- Backend A (placeholder) --------------
    def _convert_openvoice_placeholder(self, src_audio: Path, speaker_wav: Path, out_wav: Path) -> Path:
        """
        Placeholder: quando você vendorizar OpenVoice, trocamos por uma chamada real.
        Por enquanto, direciona para o mesmo fluxo de 'prosody'.
        """
        return self._convert_prosody_match(src_audio, speaker_wav, out_wav, language="pt", keep_sr=True, normalize=True)
