# pyi_rthook_tk.py
# Executa antes do seu app e ajusta env pro Tk encontrar as libs no bundle.
import os, sys
from pathlib import Path

base = Path(getattr(sys, "_MEIPASS", Path.cwd()))
# Procura diretórios tcl/tk comuns empacotados
candidatos = [
    base / "tcl8.6",
    base / "tk8.6",
    base / "tcl8.7",
    base / "tk8.7",
    base / "Resources" / "tcl8.6",
    base / "Resources" / "tk8.6",
]

tcl = next((p for p in candidatos if p.name.startswith("tcl") and p.exists()), None)
tk  = next((p for p in candidatos if p.name.startswith("tk")  and p.exists()), None)

if tcl:
    os.environ.setdefault("TCL_LIBRARY", str(tcl))
if tk:
    os.environ.setdefault("TK_LIBRARY", str(tk))
