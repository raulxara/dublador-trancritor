# pyi_rthook_env.py
# Hook executado antes do app: ajusta variáveis de ambiente críticas.
import os

# 1) Desativa o TorchScript JIT (evita necessidade de ler .py de dentro do bundle)
os.environ.setdefault("TORCH_JIT", "0")

# 2) (Opcional) fallback MPS caso Torch tente algo de aceleração em Macs
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# 3) (Opcional) evita oversubscription de threads em algumas BLAS/OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "1")
