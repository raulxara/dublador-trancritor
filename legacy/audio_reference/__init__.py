# app/__init__.py
import os

# Evita problemas de fork com GUI e libs nativas no macOS
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

# Reduz contenção/segfault por excesso de threads nativas
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

# Evita barulho de paralelismo de tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Em alguns ambientes evita erro de OpenMP duplicado
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# (Opcional; não atrapalha em Intel) deixa Torch cair para CPU quando necessário
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Garanta que o multiprocessing use spawn (seguro no macOS)
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    # já definido
    pass

# Limita threads do Torch se ele já estiver disponível
try:
    import torch  # noqa: E402
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
