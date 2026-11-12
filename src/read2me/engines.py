"""Speech engines: any mlx-audio model on the GPU, or macOS `say` as a fallback."""

from __future__ import annotations

import contextlib
import inspect
import io
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

# Short names for the models worth using. Anything containing "/" is taken as a
# Hugging Face repo id and passed straight to mlx-audio.
MODELS: dict[str, dict] = {
    "kokoro": {
        "repo": "mlx-community/Kokoro-82M-bf16",
        "voice": "af_heart",
        "max_chars": 600,
        "about": "82M, fast, 54 preset voices, blend with a comma (Apache 2.0)",
    },
    "qwen3": {
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        "voice": "ryan",
        "kwargs": {"lang_code": "english"},
        "max_chars": 300,
        "about": "1.7B, expressive, preset speakers, --instruct for style (Apache 2.0)",
    },
    "voxtral": {
        "repo": "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
        "voice": "neutral_male",
        "max_chars": 300,
        "about": "4B, Mistral, 20 preset voices (CC BY-NC 4.0: personal use)",
    },
    "chatterbox": {
        "repo": "mlx-community/chatterbox-turbo-fp16",
        "voice": None,
        "max_chars": 300,
        "about": "Resemble Chatterbox Turbo, clone a voice with --ref-audio (Apache 2.0)",
    },
}


class Engine:
    name = ""
    sample_rate = 24000
    max_chars = 400
    speed = 1.0

    def synth(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def voices(self) -> list[str]:
        return []


class MlxEngine(Engine):
    def __init__(self, model: str, voice: str | None, speed: float, extra: dict, verbose: bool):
        spec = MODELS.get(model, {"repo": model, "voice": None, "max_chars": 300})
        self.name = model
        self.repo = spec["repo"]
        self.voice = voice or spec.get("voice")
        self.speed = speed
        self.max_chars = spec.get("max_chars", 300)
        self.kwargs = {**spec.get("kwargs", {}), **extra}
        self.verbose = verbose
        self.kokoro = "kokoro" in self.repo.lower()
        with self._quiet():
            from mlx_audio.tts.utils import load_model

            self.model = load_model(self.repo)
        params = inspect.signature(self.model.generate).parameters
        self.accepts = set(params)
        self.sample_rate = int(getattr(self.model, "sample_rate", 24000))
        if speed != 1.0 and "speed" not in self.accepts:
            print(f"read2me: {model} ignores --speed", file=sys.stderr)

    def synth(self, text: str) -> np.ndarray:
        kw = dict(self.kwargs, text=text)
        if self.voice:
            kw["voice"] = self.voice
        if "speed" in self.accepts:
            kw["speed"] = self.speed
        if self.kokoro:
            kw.setdefault("lang_code", (self.voice or "a")[0])
        if "verbose" in self.accepts:
            kw["verbose"] = False
        chunks = []
        with self._quiet(stderr=True):
            for result in self.model.generate(**kw):
                chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
                self.sample_rate = int(getattr(result, "sample_rate", 0) or self.sample_rate)
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def voices(self) -> list[str]:
        if self.kokoro:
            voices = Path(_snapshot(self.repo) or ".") / "voices"
            # a = American, b = British; the others need misaki's extra language packs
            return sorted(f.stem for f in voices.glob("*.safetensors") if f.stem[0] in "ab")
        for attr in ("get_supported_speakers", "supported_speakers"):
            found = getattr(self.model, attr, None)
            if found:
                return sorted(found() if callable(found) else found)
        module = sys.modules.get(type(self.model).__module__)
        voice_map = getattr(module, "VOICE_MAP", None)
        return sorted(voice_map) if voice_map else []

    @contextlib.contextmanager
    def _quiet(self, stderr: bool = False):
        """Hide the chatter mlx-audio and its dependencies print, and some models' progress
        bars when stderr is set. Loading keeps stderr so a first-run download shows progress.
        The player writes to sys.__stdout__, so it is never caught by this."""
        if self.verbose:
            yield
            return
        with contextlib.ExitStack() as stack:
            stack.enter_context(warnings.catch_warnings())
            warnings.simplefilter("ignore")
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if stderr:
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            yield


class SayEngine(Engine):
    """macOS `say`. Instant and robotic; useful as a fallback."""

    name = "say"
    max_chars = 2000

    def __init__(self, voice: str | None, speed: float):
        self.voice = voice
        self.speed = speed

    def synth(self, text: str) -> np.ndarray:
        from scipy.io import wavfile

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.wav")
            cmd = ["say", "-o", out, "--file-format=WAVE", f"--data-format=LEI16@{self.sample_rate}"]
            cmd += ["-r", str(int(185 * self.speed))]
            if self.voice:
                cmd += ["-v", self.voice]
            subprocess.run(cmd + ["--", text], check=True)
            _, data = wavfile.read(out)
        return data.astype(np.float32) / 32768.0

    def voices(self) -> list[str]:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
        return [line.split("  ")[0].strip() for line in listing.splitlines() if " en_" in line]


def load(model: str, voice: str | None, speed: float, extra: dict, verbose: bool) -> Engine:
    if model == "say":
        return SayEngine(voice, speed)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("read2me: neural voices need an Apple Silicon Mac (MLX); use -m say")
    if model not in MODELS and "/" not in model:
        raise SystemExit(f"read2me: unknown model {model!r}; try one of {', '.join([*MODELS, 'say'])} or a Hugging Face repo id")
    repo = MODELS.get(model, {}).get("repo", model)
    if _snapshot(repo):
        # Already downloaded: skip the Hub entirely, so startup is fast and works offline.
        # Must be set before huggingface_hub is first imported, which reads it once.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    else:
        print(f"read2me: downloading {repo} (first run only)…", file=sys.stderr)
    _quiet_hub()
    _shorten_espeak_data_path()
    return MlxEngine(model, voice, speed, extra, verbose)


def _shorten_espeak_data_path() -> None:
    """espeak-ng keeps its data path in a 160-byte buffer and silently falls back to a
    path baked in at build time when it's longer, then exits the whole process on the
    first unknown word. Deep install locations hit that, so point it at a short copy (not a
    symlink: phonemizer resolves symlinks back to the long path)."""
    try:
        import espeakng_loader
        import misaki.espeak  # noqa: F401 (sets espeak's library and data path on import)
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError:
        return
    data = espeakng_loader.get_data_path()
    if len(data.encode()) < 150:
        return
    copy = Path.home() / ".cache" / "read2me" / "espeak-ng-data"
    source = copy / ".source"
    if not source.exists() or source.read_text() != data:
        if copy.is_symlink() or copy.is_file():
            copy.unlink()
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(data, copy)
        source.write_text(data)
    EspeakWrapper.set_data_path(str(copy))


def _snapshot(repo: str) -> str | None:
    """The newest cached snapshot of a Hugging Face repo, without importing the Hub client."""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    snapshots = Path(hub) / f"models--{repo.replace('/', '--')}" / "snapshots"
    found = sorted(snapshots.glob("*/config.json"), key=lambda p: p.stat().st_mtime)
    return str(found[-1].parent) if found else None


def _quiet_hub() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # transformers is only used for tokenizers here; without these it warns that PyTorch is missing
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    # misaki falls back to espeak for unknown words and the phonemizer logs a warning each time
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    try:
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.set_verbosity_error()
    except Exception:
        pass
