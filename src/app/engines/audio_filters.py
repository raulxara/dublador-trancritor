import subprocess
from pathlib import Path


class AudioFilters:
    @staticmethod
    def tempo(value):
        filters = []
        while value < 0.5:
            filters.append("atempo=0.5")
            value /= 0.5
        while value > 2:
            filters.append("atempo=2")
            value /= 2
        filters.append(f"atempo={value:.8f}")
        return filters

    @classmethod
    def convert(cls, source: Path, target: Path, speed=1.0, pitch=0.0, duration=None):
        factor = 2 ** (pitch / 12)
        filters = [f"asetrate={24000 * factor:.8f}", "aresample=24000"] + cls.tempo(speed / factor)
        if duration is not None:
            filters += ["apad", f"atrim=duration={duration:.6f}"]
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-af",
                ",".join(filters),
                "-ar",
                "24000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(target),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
