from pathlib import Path
import sys
import os

APP_NAME = "Dublador Secret Brand World"

# Detecta se está rodando empacotado (PyInstaller) ou em dev
IS_FROZEN = getattr(sys, "frozen", False)

if IS_FROZEN:
    # Base de leitura (recursos empacotados)
    BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    # Base de escrita (dados do usuário)
    if sys.platform == "darwin":
        DATA_ROOT = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        DATA_ROOT = Path.home() / f".{APP_NAME.replace(' ', '_').lower()}"
else:
    # Em desenvolvimento: use a pasta do projeto
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_ROOT = BASE_DIR / "data"

# Diretórios principais
VOICES_DIR    = DATA_ROOT / "voices"
LOGS_DIR      = DATA_ROOT / "logs"
PROJECTS_DIR  = DATA_ROOT / "projects"
ASSETS_DIR    = BASE_DIR / "assets"
for d in (VOICES_DIR, LOGS_DIR, PROJECTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ====== Áudio: taxas de amostragem ======
# ASR (reconhecimento): 16 kHz mono é o padrão ideal
SAMPLE_RATE_ASR = 16000
# TTS (Coqui XTTS normalmente trabalha em 22.05 kHz)
SAMPLE_RATE_TTS = 22050
# Compat: vários módulos importam SAMPLE_RATE -> usamos do TTS
SAMPLE_RATE = SAMPLE_RATE_TTS

# ====== Defaults usados na UI ======
LANG_DEFAULT       = "pt"
EXPORT_MP3_DEFAULT = True
MP3_BITRATE        = "192k"

# ====== Parâmetros de validação de voz-base (USADOS pelo validator) ======
# Duração mínima e máxima aceitáveis para o áudio-base
MIN_VOICE_SECONDS = 3.0     # mínimo absoluto para XTTS entender (ideal >= 6–10s)
MAX_VOICE_SECONDS = 180.0   # limite superior para evitar arquivos enormes

# Limites de qualidade (o validator costuma checar estes)
VOICE_SILENCE_RATIO_MAX = 0.25   # fração máxima de silêncio (0.00–1.00)
VOICE_CLIP_RATIO_MAX    = 0.02   # fração máxima com clipping (0.00–1.00)

# ====== ASR (Whisper-PyTorch) ======
# Tamanho do modelo Whisper. Opções: "tiny", "base", "small", "medium", "large".
# Para CPU, "tiny" é o mais leve e foi o usado nos seus testes.
ASR_MODEL_SIZE = os.getenv("DUBBER_ASR_MODEL", "tiny")

# Dispositivo para o Whisper. No seu Mac: "cpu".
ASR_DEVICE = os.getenv("DUBBER_ASR_DEVICE", "cpu")
