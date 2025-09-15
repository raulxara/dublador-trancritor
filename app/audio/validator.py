from pathlib import Path
from typing import Dict
from app.config import MIN_VOICE_SECONDS, MAX_VOICE_SECONDS
from app.audio.utils import measure_audio_stats

def validate_voice_sample(clean_wav: Path) -> Dict:
    stats = measure_audio_stats(clean_wav)

    ok_duration = (stats["duration_sec"] >= MIN_VOICE_SECONDS) and (stats["duration_sec"] <= MAX_VOICE_SECONDS)
    ok_silence  = stats["silence_ratio"] <= 0.6   # muita pausa? > 60% é ruim
    ok_clipping = stats["clip_ratio"] <= 0.001    # >0.1% indica saturação/clipping
    ok_peak     = stats["peak"] <= 0.999          # se bate no teto, é suspeito

    passed = all([ok_duration, ok_silence, ok_clipping, ok_peak])

    tips = []
    if not ok_duration:
        tips.append(f"Duração deve ficar entre {MIN_VOICE_SECONDS:.0f}s e {MAX_VOICE_SECONDS:.0f}s.")
    if not ok_silence:
        tips.append("Muito silêncio; grave em ambiente mais silencioso e fale de forma contínua.")
    if not ok_clipping or not ok_peak:
        tips.append("Picos muito altos (clipping); abaixe o ganho de entrada ou afaste o microfone.")

    return {
        "passed": passed,
        "stats": stats,
        "checks": {
            "duration_ok": ok_duration,
            "silence_ok": ok_silence,
            "clipping_ok": ok_clipping,
            "peak_ok": ok_peak
        },
        "tips": tips
    }
