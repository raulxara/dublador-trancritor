import json, shutil, uuid, subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional

from app.config import VOICES_DIR
from app.audio.utils import sniff_media_type, ensure_wav_mono_22050
from app.audio.validator import validate_voice_sample

@dataclass
class BaseVoice:
    id: str
    name: str
    # arquivos
    raw_path: str
    clean_wav: str
    # resultados de validação
    validation: Dict

class VoiceManager:
    def __init__(self, storage_dir: Path = VOICES_DIR):
        self.dir = storage_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _voice_dir(self, vid: str) -> Path:
        return self.dir / vid

    def list_voices(self) -> List[BaseVoice]:
        voices: List[BaseVoice] = []
        for child in self.dir.iterdir():
            if not child.is_dir(): 
                continue
            meta = child / "voice.json"
            if meta.exists():
                try:
                    j = json.loads(meta.read_text(encoding="utf-8"))
                    voices.append(BaseVoice(**j))
                except Exception:
                    pass
        return voices

    def get_voice(self, vid: str) -> Optional[BaseVoice]:
        meta = self._voice_dir(vid) / "voice.json"
        if not meta.exists():
            return None
        j = json.loads(meta.read_text(encoding="utf-8"))
        return BaseVoice(**j)

    def add_voice_from_file(self, src_path: Path, display_name: Optional[str] = None) -> BaseVoice:
        if not src_path.exists():
            raise FileNotFoundError(str(src_path))

        media = sniff_media_type(src_path)
        if media == "unknown":
            raise ValueError(f"Formato não suportado: {src_path.suffix}")

        vid = uuid.uuid4().hex[:8]
        vdir = self._voice_dir(vid)
        vdir.mkdir(parents=True, exist_ok=True)

        # salvar original
        raw_dst = vdir / f"raw{src_path.suffix.lower()}"
        shutil.copy2(src_path, raw_dst)

        # padronizar para clean.wav
        clean_wav = vdir / "clean.wav"
        ensure_wav_mono_22050(raw_dst, clean_wav)

        # validar
        validation = validate_voice_sample(clean_wav)

        voice = BaseVoice(
            id=vid,
            name=display_name or f"Voz {vid}",
            raw_path=str(raw_dst),
            clean_wav=str(clean_wav),
            validation=validation
        )
        (vdir / "voice.json").write_text(json.dumps(asdict(voice), indent=2, ensure_ascii=False), encoding="utf-8")
        return voice

    def play_preview(self, vid: str) -> bool:
        voice = self.get_voice(vid)
        if not voice:
            return False
        # macOS tem 'afplay' nativo
        try:
            subprocess.Popen(["afplay", voice.clean_wav])
            return True
        except Exception:
            return False
