import threading
import multiprocessing as mp
from pathlib import Path
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
import subprocess

from app.config import (
    VOICES_DIR, LANG_DEFAULT, LOGS_DIR,
    EXPORT_MP3_DEFAULT, MP3_BITRATE
)
from app.voice_manager import VoiceManager, BaseVoice
from app.engines.tts_xtts import XTTSEngine
# ASR (estável em mac Intel): Whisper PyTorch
from app.engines.asr_openai import ASREngine

from app.utils.projects import new_job_dir
from app.audio.utils import ensure_wav_mono_16000
from app.audio.post import apply_speed_pitch, wav_to_mp3  # sem stretch_to_duration
from app.engines.vc_s2s import VCEngine


# ========= subprocesso para S2S (isola libs nativas e evita crash no processo principal) =========
def _vc_convert_child(src: str, speaker: str, out_wav: str, language: str):
    """Roda a conversão voz->voz em outro processo (spawn)."""
    try:
        from pathlib import Path as _Path
        from app.engines.vc_s2s import VCEngine as _VCEngine
        vc = _VCEngine.instance()
        vc.convert(
            src_audio=_Path(src),
            speaker_wav=_Path(speaker),
            out_wav=_Path(out_wav),
            language=language,
            keep_sr=True,
            normalize=True
        )
    except Exception as e:
        # registra stacktrace do filho
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            (_Path(LOGS_DIR) / "s2s_child.log").write_text(str(e), encoding="utf-8")
        except Exception:
            pass
        raise


class DubberApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Dublador Secret Brand World")
        self.geometry("1100x760")

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("green")

        self.vm = VoiceManager()
        self.xtts = None   # carregado sob demanda
        self.asr = None    # carregado sob demanda

        # Guarda o último job da aba Áudio→Voz (para reaproveitar pasta)
        self.asr_current_job_dir = None

        self._build_ui()
        self._refresh_voice_list()
        self._refresh_voice_dropdowns()

        self.last_out = None
        self.last_dir = None

    # =============== UI ===============
    def _build_ui(self):
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=12)

        self.tab_voices = self.tabs.add("Vozes")
        self.tab_tts = self.tabs.add("Dublagem (Texto → Voz)")
        self.tab_asrtts = self.tabs.add("Dublagem (Áudio → Voz)")

        # ----------- Tab: Vozes -----------
        header = ctk.CTkFrame(self.tab_voices)
        header.pack(fill="x", padx=12, pady=(12, 6))

        title = ctk.CTkLabel(header, text="Vozes Base", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(side="left", padx=8, pady=8)

        btn_add = ctk.CTkButton(header, text="+ Adicionar Voz Base", command=self._on_add_voice)
        btn_add.pack(side="right", padx=8, pady=8)

        body = ctk.CTkFrame(self.tab_voices)
        body.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        self.list_container = ctk.CTkScrollableFrame(body)
        self.list_container.pack(fill="both", expand=True, padx=8, pady=8)

        footer = ctk.CTkFrame(self.tab_voices)
        footer.pack(fill="x", padx=12, pady=(0, 12))
        info = ctk.CTkLabel(footer, text=f"Pasta de vozes: {VOICES_DIR}")
        info.pack(side="left", padx=8, pady=8)

        # -------- Tab: Dublagem (Texto → Voz) -----
        wrap = ctk.CTkFrame(self.tab_tts)
        wrap.pack(fill="both", expand=True, padx=12, pady=12)

        # linha 1 — seleção de voz e idioma
        row1 = ctk.CTkFrame(wrap)
        row1.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(row1, text="Voz base:").pack(side="left", padx=(0, 8))
        self.voice_choice_tts = tk.StringVar(value="")
        self.voice_select_tts = ctk.CTkOptionMenu(row1, variable=self.voice_choice_tts, values=[], width=280)
        self.voice_select_tts.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(row1, text="Idioma:").pack(side="left", padx=(0, 8))
        self.lang_var_tts = tk.StringVar(value=LANG_DEFAULT)
        ctk.CTkEntry(row1, textvariable=self.lang_var_tts, width=70).pack(side="left", padx=(0, 20))

        # linha 2 — controles de speed/pitch/mp3
        row1b = ctk.CTkFrame(wrap)
        row1b.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(row1b, text="Velocidade (0.5–1.5):").pack(side="left", padx=(0, 8))
        self.speed_var_tts = tk.StringVar(value="1.0")
        ctk.CTkEntry(row1b, textvariable=self.speed_var_tts, width=70).pack(side="left", padx=(0, 20))

        ctk.CTkLabel(row1b, text="Pitch (semitons -6..+6):").pack(side="left", padx=(0, 8))
        self.pitch_var_tts = tk.StringVar(value="0")
        ctk.CTkEntry(row1b, textvariable=self.pitch_var_tts, width=70).pack(side="left", padx=(0, 20))

        self.mp3_var_tts = tk.BooleanVar(value=EXPORT_MP3_DEFAULT)
        ctk.CTkCheckBox(row1b, text=f"Salvar MP3 ({MP3_BITRATE})", variable=self.mp3_var_tts).pack(side="left")

        # texto
        row2 = ctk.CTkFrame(wrap)
        row2.pack(fill="both", expand=True)
        ctk.CTkLabel(row2, text="Texto para dublar:").pack(anchor="w", pady=(0, 6))
        self.tts_text = ctk.CTkTextbox(row2, height=220)
        self.tts_text.pack(fill="both", expand=True)

        # botões
        row3 = ctk.CTkFrame(wrap)
        row3.pack(fill="x", pady=10)
        self.btn_gen = ctk.CTkButton(row3, text="🎙️ Gerar Áudio (TTS)", command=self._on_generate_tts)
        self.btn_gen.pack(side="left")
        self.btn_play = ctk.CTkButton(row3, text="▶ Preview último", state="disabled", command=self._on_play_last)
        self.btn_play.pack(side="left", padx=8)
        self.btn_open = ctk.CTkButton(row3, text="📂 Abrir pasta", state="disabled", command=self._on_open_last_dir)
        self.btn_open.pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="Pronto.")
        ctk.CTkLabel(wrap, textvariable=self.status_var).pack(anchor="w", pady=(8, 0))

        # -------- Tab: Dublagem (Áudio → Voz) -----
        aw = ctk.CTkFrame(self.tab_asrtts)
        aw.pack(fill="both", expand=True, padx=12, pady=12)

        # linha A — voz + arquivo
        rowa1 = ctk.CTkFrame(aw)
        rowa1.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(rowa1, text="Voz base:").pack(side="left", padx=(0, 8))
        self.voice_choice_asr = tk.StringVar(value="")
        self.voice_select_asr = ctk.CTkOptionMenu(rowa1, variable=self.voice_choice_asr, values=[], width=280)
        self.voice_select_asr.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(rowa1, text="Arquivo (áudio/vídeo):").pack(side="left", padx=(0, 8))
        self.asr_src_path_var = tk.StringVar(value="")
        ctk.CTkEntry(rowa1, textvariable=self.asr_src_path_var, width=360).pack(side="left", padx=(0, 8))
        ctk.CTkButton(rowa1, text="Escolher…", command=self._on_pick_asr_file).pack(side="left")

        # linha B — Modo + idioma + speed/pitch/mp3
        rowa2 = ctk.CTkFrame(aw)
        rowa2.pack(fill="x", pady=(0, 10))

        self.mode_var = tk.StringVar(value="S2S (tempo idêntico)")
        ctk.CTkLabel(rowa2, text="Modo:").pack(side="left", padx=(0, 8))
        ctk.CTkOptionMenu(rowa2, variable=self.mode_var, values=["S2S (tempo idêntico)", "TTS"]).pack(side="left", padx=(0, 20))

        ctk.CTkLabel(rowa2, text="Idioma destino (TTS):").pack(side="left", padx=(0, 8))
        self.lang_var_asrtts = tk.StringVar(value=LANG_DEFAULT)
        ctk.CTkEntry(rowa2, textvariable=self.lang_var_asrtts, width=70).pack(side="left", padx=(0, 20))

        ctk.CTkLabel(rowa2, text="Velocidade (0.5–1.5):").pack(side="left", padx=(0, 8))
        self.speed_var_asr = tk.StringVar(value="1.0")
        ctk.CTkEntry(rowa2, textvariable=self.speed_var_asr, width=70).pack(side="left", padx=(0, 20))

        ctk.CTkLabel(rowa2, text="Pitch (semitons -6..+6):").pack(side="left", padx=(0, 8))
        self.pitch_var_asr = tk.StringVar(value="0")
        ctk.CTkEntry(rowa2, textvariable=self.pitch_var_asr, width=70).pack(side="left", padx=(0, 20))

        self.mp3_var_asr = tk.BooleanVar(value=EXPORT_MP3_DEFAULT)
        ctk.CTkCheckBox(rowa2, text=f"Salvar MP3 ({MP3_BITRATE})", variable=self.mp3_var_asr).pack(side="left")

        # linha C — caixa de texto (transcrição editável)
        rowa3 = ctk.CTkFrame(aw)
        rowa3.pack(fill="both", expand=True)
        ctk.CTkLabel(rowa3, text="Transcrição (edite antes de gerar):").pack(anchor="w", pady=(0, 6))
        self.asr_text_box = ctk.CTkTextbox(rowa3, height=220)
        self.asr_text_box.pack(fill="both", expand=True)

        # linha D — botões separados
        rowa4 = ctk.CTkFrame(aw)
        rowa4.pack(fill="x", pady=10)
        self.btn_transcribe = ctk.CTkButton(rowa4, text="📝 Transcrever", command=self._on_transcribe_only)
        self.btn_transcribe.pack(side="left")
        self.btn_generate_from_text = ctk.CTkButton(rowa4, text="🎙️ Gerar dublagem", command=self._on_generate_asrtts, state="disabled")
        self.btn_generate_from_text.pack(side="left", padx=8)

        self.btn_play2 = ctk.CTkButton(rowa4, text="▶ Preview último", state="disabled", command=self._on_play_last)
        self.btn_play2.pack(side="left", padx=8)
        self.btn_open2 = ctk.CTkButton(rowa4, text="📂 Abrir pasta", state="disabled", command=self._on_open_last_dir)
        self.btn_open2.pack(side="left", padx=8)

        self.status_var2 = tk.StringVar(value="Pronto.")
        ctk.CTkLabel(aw, textvariable=self.status_var2).pack(anchor="w", pady=(8, 0))

    def _on_generate_asrtts(self):
        mode = (self.mode_var.get() or "TTS").lower()
        if mode.startswith("s2s"):
            self._on_convert_s2s()
        else:
            self._on_generate_from_text()

    # =============== Vozes ===============
    def _refresh_voice_list(self):
        for w in self.list_container.winfo_children():
            w.destroy()

        voices = self.vm.list_voices()
        if not voices:
            ctk.CTkLabel(self.list_container, text="Nenhuma voz base ainda. Clique em “+ Adicionar Voz Base”.").pack(pady=16)
            return

        for v in voices:
            self._render_voice_item(v)

    def _render_voice_item(self, voice: BaseVoice):
        card = ctk.CTkFrame(self.list_container)
        card.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(card, text=f"{voice.name}  (id: {voice.id})", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=(10, 2))

        val = voice.validation
        status = "APROVADA" if val.get("passed") else "COM PENDÊNCIAS"
        stats = val.get("stats", {})
        line = f"Status: {status}  |  duração: {stats.get('duration_sec','?')}s  |  silêncio: {stats.get('silence_ratio','?')}  |  clipping: {stats.get('clip_ratio','?')}"
        ctk.CTkLabel(card, text=line).pack(anchor="w", padx=10, pady=(0, 6))

        tips = val.get("tips") or []
        if tips:
            tipbox = ctk.CTkTextbox(card, height=60)
            tipbox.pack(fill="x", padx=10, pady=(0, 8))
            tipbox.insert("1.0", "Sugestões:\n- " + "\n- ".join(tips))
            tipbox.configure(state="disabled")

        btn_row = ctk.CTkFrame(card)
        btn_row.pack(fill="x", padx=10, pady=(0, 10))

        def on_preview():
            ok = self.vm.play_preview(voice.id)
            if not ok:
                messagebox.showerror("Erro", "Falha ao tocar preview. Verifique o áudio.")

        ctk.CTkButton(btn_row, text="▶ Preview", command=on_preview).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(card, text=f"Arquivos: raw={voice.raw_path}  |  clean={voice.clean_wav}").pack(anchor="w", padx=10, pady=(0, 10))

    def _refresh_voice_dropdowns(self):
        def make_choices():
            mapping = {}
            vals = []
            for v in self.vm.list_voices():
                label = f"{v.name} ({v.id})"
                mapping[label] = v.id
                vals.append(label)
            if not vals:
                vals = ["— sem vozes —"]
            return mapping, vals

        self.voice_name_by_id_tts, choices1 = make_choices()
        self.voice_select_tts.configure(values=choices1)
        self.voice_choice_tts.set(choices1[0])

        self.voice_name_by_id_asr, choices2 = make_choices()
        self.voice_select_asr.configure(values=choices2)
        self.voice_choice_asr.set(choices2[0])

    def _on_add_voice(self):
        fpath = filedialog.askopenfilename(
            title="Escolher arquivo de voz (wav/mp3/mp4...)",
            filetypes=[("Áudio/Vídeo", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.mp4 *.mov"), ("Todos", "*.*")]
        )
        if not fpath:
            return

        name = simpledialog.askstring("Nome da voz", "Digite um nome para essa voz base:")
        if not name:
            name = "Voz"

        def worker():
            try:
                voice = self.vm.add_voice_from_file(Path(fpath), display_name=name)
                self.after(0, lambda: (
                    messagebox.showinfo("OK", f"Voz adicionada: {voice.name}"),
                    self._refresh_voice_list(),
                    self._refresh_voice_dropdowns()
                ))
            except Exception as e:
                import traceback, sys
                tb = traceback.format_exc(); print(tb, file=sys.stderr)
                try:
                    LOGS_DIR.mkdir(parents=True, exist_ok=True)
                    (LOGS_DIR / "runtime.log").write_text(tb, encoding="utf-8")
                except Exception:
                    pass
                self.after(0, lambda err=e: messagebox.showerror("Erro", str(err)))

        threading.Thread(target=worker, daemon=True).start()

    # =============== Helpers ===============
    def _parse_speed_pitch(self, speed_str: str, pitch_str: str):
        # speed em [0.5, 1.5], pitch em [-12, +12] (usaremos -6..+6 na UI)
        try:
            speed = float(speed_str.strip())
        except Exception:
            speed = 1.0
        speed = max(0.5, min(1.5, speed))

        try:
            semitones = int(pitch_str.strip())
        except Exception:
            semitones = 0
        semitones = max(-12, min(12, semitones))
        return speed, semitones

    # =============== TTS (Texto→Voz) ===============
    def _on_generate_tts(self):
        label = self.voice_choice_tts.get()
        vid = self.voice_name_by_id_tts.get(label)
        if not vid:
            messagebox.showwarning("Atenção", "Adicione e selecione uma voz base na aba 'Vozes'.")
            return

        voice = self.vm.get_voice(vid)
        if not voice:
            messagebox.showerror("Erro", "Voz não encontrada.")
            return

        text = self.tts_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Atenção", "Digite um texto para dublar.")
            return

        lang = self.lang_var_tts.get().strip() or LANG_DEFAULT
        speed, semitones = self._parse_speed_pitch(self.speed_var_tts.get(), self.pitch_var_tts.get())
        save_mp3 = bool(self.mp3_var_tts.get())

        self.btn_gen.configure(state="disabled")
        self.status_var.set("Gerando áudio...")

        def worker():
            try:
                if self.xtts is None:
                    self.xtts = XTTSEngine.instance()

                job_dir = new_job_dir(prefix="tts")
                raw_path = job_dir / "raw.wav"
                final_wav = job_dir / "tts.wav"

                # 1) síntese base (modo 'smart' que limpa pontuação final)
                self.xtts.synthesize_smart_to_file(text, Path(voice.clean_wav), lang, raw_path, pause_ms=180)

                # 2) pós-processamento (speed/pitch — opcional)
                if abs(speed - 1.0) > 1e-6 or semitones != 0:
                    apply_speed_pitch(raw_path, final_wav, speed=speed, semitones=semitones)
                else:
                    raw_path.rename(final_wav)

                # 3) MP3 opcional
                if save_mp3:
                    wav_to_mp3(final_wav, job_dir / "tts.mp3")

                def done():
                    self.last_out = final_wav
                    self.last_dir = job_dir
                    self.btn_play.configure(state="normal")
                    self.btn_open.configure(state="normal")
                    self.btn_gen.configure(state="normal")
                    self.status_var.set(f"Áudio gerado: {final_wav.name}{' (+ MP3)' if save_mp3 else ''}")
                    try:
                        subprocess.Popen(["afplay", str(final_wav)])
                    except Exception:
                        pass
                self.after(0, done)

            except Exception as e:
                import traceback, sys
                tb = traceback.format_exc(); print(tb, file=sys.stderr)
                try:
                    LOGS_DIR.mkdir(parents=True, exist_ok=True)
                    (LOGS_DIR / "runtime.log").write_text(tb, encoding="utf-8")
                except Exception:
                    pass
                self.after(0, lambda err=e: (
                    self.btn_gen.configure(state="normal"),
                    self.status_var.set("Erro ao gerar."),
                    messagebox.showerror("Erro", str(err))
                ))

        threading.Thread(target=worker, daemon=True).start()

    # =============== Áudio→Voz: Transcrever / Gerar ===============
    def _on_pick_asr_file(self):
        fpath = filedialog.askopenfilename(
            title="Escolher arquivo (áudio/vídeo)",
            filetypes=[("Áudio/Vídeo", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.mp4 *.mov"), ("Todos", "*.*")]
        )
        if fpath:
            self.asr_src_path_var.set(fpath)

    def _on_transcribe_only(self):
        src = self.asr_src_path_var.get().strip()
        if not src or not Path(src).exists():
            messagebox.showwarning("Atenção", "Escolha um arquivo de origem válido.")
            return

        self.btn_transcribe.configure(state="disabled")
        self.btn_generate_from_text.configure(state="disabled")
        self.status_var2.set("Carregando modelo ASR (pode demorar na primeira vez)...")

        def worker():
            try:
                if self.asr is None:
                    self.asr = ASREngine.instance()

                # prepara job e converte p/ 16 kHz mono (padrão bom p/ ASR)
                src_path = Path(src)
                job_dir = new_job_dir(prefix="asr-tts")
                tmp_src = job_dir / "source.wav"

                self.after(0, lambda: self.status_var2.set("Preparando áudio (16 kHz, mono)..."))
                ensure_wav_mono_16000(src_path, tmp_src)

                # transcrever
                self.after(0, lambda: self.status_var2.set("Transcrevendo..."))
                result = self.asr.transcribe(tmp_src)
                text = (result.get("text") or "").strip()

                def done_tx():
                    self.asr_current_job_dir = job_dir
                    self.asr_text_box.delete("1.0", "end")
                    self.asr_text_box.insert("1.0", text)
                    self.btn_transcribe.configure(state="normal")
                    self.btn_generate_from_text.configure(state="normal" if text else "disabled")
                    self.status_var2.set("Transcrição pronta. Revise/edite o texto e clique em “Gerar dublagem”.")
                self.after(0, done_tx)

            except Exception as e:
                import traceback, sys
                tb = traceback.format_exc(); print(tb, file=sys.stderr)
                try:
                    LOGS_DIR.mkdir(parents=True, exist_ok=True)
                    (LOGS_DIR / "runtime.log").write_text(tb, encoding="utf-8")
                except Exception:
                    pass
                self.after(0, lambda err=e: (
                    self.btn_transcribe.configure(state="normal"),
                    self.btn_generate_from_text.configure(state="disabled"),
                    self.status_var2.set("Erro ao transcrever."),
                    messagebox.showerror("Erro", str(err))
                ))

        threading.Thread(target=worker, daemon=True).start()

    def _on_generate_from_text(self):
        label = self.voice_choice_asr.get()
        vid = self.voice_name_by_id_asr.get(label)
        if not vid:
            messagebox.showwarning("Atenção", "Adicione e selecione uma voz base na aba 'Vozes'.")
            return

        voice = self.vm.get_voice(vid)
        if not voice:
            messagebox.showerror("Erro", "Voz não encontrada.")
            return

        text = self.asr_text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Atenção", "Não há texto para gerar.")
            return

        lang_tts = self.lang_var_asrtts.get().strip() or LANG_DEFAULT
        speed, semitones = self._parse_speed_pitch(self.speed_var_asr.get(), self.pitch_var_asr.get())
        save_mp3 = bool(self.mp3_var_asr.get())

        self.btn_generate_from_text.configure(state="disabled")
        self.status_var2.set("Gerando dublagem com TTS...")

        def worker():
            try:
                if self.xtts is None:
                    self.xtts = XTTSEngine.instance()

                job_dir = self.asr_current_job_dir or new_job_dir(prefix="asr-tts")
                raw_path = job_dir / "raw.wav"
                final_wav = job_dir / "dubbing.wav"

                # 1) síntese base (modo smart recomendado)
                self.xtts.synthesize_smart_to_file(text, Path(voice.clean_wav), lang_tts, raw_path, pause_ms=180)

                # 2) pós-processamento
                if abs(speed - 1.0) > 1e-6 or semitones != 0:
                    apply_speed_pitch(raw_path, final_wav, speed=speed, semitones=semitones)
                else:
                    raw_path.rename(final_wav)

                # 3) MP3 opcional
                if save_mp3:
                    wav_to_mp3(final_wav, job_dir / "dubbing.mp3")

                # 4) salva texto
                (job_dir / "transcript.txt").write_text(text, encoding="utf-8")

                def done():
                    self.last_out = final_wav
                    self.last_dir = job_dir
                    self.btn_play2.configure(state="normal"); self.btn_open2.configure(state="normal")
                    self.btn_play.configure(state="normal");  self.btn_open.configure(state="normal")
                    self.btn_generate_from_text.configure(state="normal")
                    self.status_var2.set(f"Dublagem gerada: {final_wav.name}{' (+ MP3)' if save_mp3 else ''}")
                    try:
                        subprocess.Popen(["afplay", str(final_wav)])
                    except Exception:
                        pass
                self.after(0, done)

            except Exception as e:
                import traceback, sys
                tb = traceback.format_exc(); print(tb, file=sys.stderr)
                try:
                    LOGS_DIR.mkdir(parents=True, exist_ok=True)
                    (LOGS_DIR / "runtime.log").write_text(tb, encoding="utf-8")
                except Exception:
                    pass
                self.after(0, lambda err=e: (
                    self.btn_generate_from_text.configure(state="normal"),
                    self.status_var2.set("Erro ao gerar."),
                    messagebox.showerror("Erro", str(err))
                ))

        threading.Thread(target=worker, daemon=True).start()

    def _on_convert_s2s(self):
        # usa o arquivo de origem e a voz-base selecionada; ignora texto
        label = self.voice_choice_asr.get()
        vid = self.voice_name_by_id_asr.get(label)
        if not vid:
            messagebox.showwarning("Atenção", "Adicione e selecione uma voz base na aba 'Vozes'.")
            return
        voice = self.vm.get_voice(vid)
        if not voice:
            messagebox.showerror("Erro", "Voz não encontrada.")
            return

        src = self.asr_src_path_var.get().strip()
        if not src or not Path(src).exists():
            messagebox.showwarning("Atenção", "Escolha um arquivo de origem válido.")
            return

        save_mp3 = bool(self.mp3_var_asr.get())
        lang_tts = self.lang_var_asrtts.get().strip() or LANG_DEFAULT  # XTTS precisa de lang, mas VC usa pouco aqui

        self.btn_generate_from_text.configure(state="disabled")
        self.status_var2.set("Convertendo voz (S2S)…")

        def worker():
            try:
                job_dir = self.asr_current_job_dir or new_job_dir(prefix="asr-tts")
                final_wav = job_dir / "dubbing.wav"

                # roda a conversão em subprocesso "spawn"
                ctx = mp.get_context("spawn")
                p = ctx.Process(
                    target=_vc_convert_child,
                    args=(str(src), str(voice.clean_wav), str(final_wav), lang_tts),
                    daemon=False,
                )
                p.start()
                p.join()

                if p.exitcode != 0:
                    raise RuntimeError(f"Conversão S2S falhou (exitcode={p.exitcode}). Veja logs em {LOGS_DIR}.")

                # MP3 opcional
                if save_mp3:
                    wav_to_mp3(final_wav, job_dir / "dubbing.mp3")

                def done():
                    self.last_out = final_wav
                    self.last_dir = job_dir
                    self.btn_play2.configure(state="normal"); self.btn_open2.configure(state="normal")
                    self.btn_play.configure(state="normal");  self.btn_open.configure(state="normal")
                    self.btn_generate_from_text.configure(state="normal")
                    self.status_var2.set(f"Dublagem S2S gerada: {final_wav.name}{' (+ MP3)' if save_mp3 else ''}")
                    try:
                        subprocess.Popen(["afplay", str(final_wav)])
                    except Exception:
                        pass
                self.after(0, done)

            except Exception as e:
                import traceback, sys
                tb = traceback.format_exc(); print(tb, file=sys.stderr)
                try:
                    LOGS_DIR.mkdir(parents=True, exist_ok=True)
                    (LOGS_DIR / "runtime.log").write_text(tb, encoding="utf-8")
                except Exception:
                    pass
                self.after(0, lambda err=e: (
                    self.btn_generate_from_text.configure(state="normal"),
                    self.status_var2.set("Erro na conversão S2S."),
                    messagebox.showerror("Erro", str(err))
                ))

        threading.Thread(target=worker, daemon=True).start()

    # =============== Utilidades comuns ===============
    def _on_play_last(self):
        if not self.last_out or not Path(self.last_out).exists():
            messagebox.showwarning("Atenção", "Nenhum áudio recente para tocar.")
            return
        try:
            subprocess.Popen(["afplay", str(self.last_out)])
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _on_open_last_dir(self):
        if not self.last_dir or not Path(self.last_dir).exists():
            messagebox.showwarning("Atenção", "Nenhuma pasta para abrir.")
            return
        subprocess.Popen(["open", str(self.last_dir)])


def main():
    app = DubberApp()
    app.mainloop()


if __name__ == "__main__":
    main()
