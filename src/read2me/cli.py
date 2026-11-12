"""read2me: read a markdown document aloud with a local neural voice."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "read2me" / "config.toml"

DEFAULTS = {"model": "kokoro", "voice": None, "speed": 1.0, "code": "announce"}

EPILOG = f"""\
examples:
  read2me openspec/changes/foo/proposal.md
  read2me design.md --from Decisions          start at the first heading matching "Decisions"
  read2me design.md -v bm_george -s 1.15      British voice, a little faster
  read2me design.md -v af_heart,af_bella      blend two Kokoro voices
  read2me design.md -m qwen3 --instruct "calm, measured narration"
  read2me design.md -o design.m4a             save audio instead of playing
  read2me design.md --print                   show the text that would be spoken
  read2me --list-voices [-m MODEL]

defaults live in {CONFIG}, e.g.
  model = "kokoro"
  voice = "af_heart"
  speed = 1.1

while playing: space pause, ←/→ previous/next segment, ↑/↓ previous/next section,
+/- speed (kokoro), q quit.
"""


def main(argv: list[str] | None = None) -> None:
    config = {**DEFAULTS, **_read_config()}
    p = argparse.ArgumentParser(
        prog="read2me",
        description="Read a markdown document aloud with a local text-to-speech model.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("file", nargs="?", help="markdown file, or - for stdin")
    p.add_argument("-m", "--model", default=config["model"], help="kokoro (default), qwen3, voxtral, chatterbox, say, or a Hugging Face repo id")
    p.add_argument("-v", "--voice", default=config["voice"], help="voice name; for kokoro, comma-separate to blend")
    p.add_argument("-s", "--speed", type=float, default=config["speed"], help="0.5 to 2.0 (default %(default)s)")
    p.add_argument("--from", dest="start", metavar="HEADING", help="start at the first heading containing this text")
    p.add_argument("--code", choices=["announce", "read", "skip"], default=config["code"], help="what to do with code blocks (default %(default)s)")
    p.add_argument("--instruct", default=config.get("instruct"), help="style instruction, for models that take one (qwen3)")
    p.add_argument("--ref-audio", default=config.get("ref_audio"), help="reference clip to clone a voice from, for models that support it")
    p.add_argument("-o", "--output", type=Path, help="write audio to this .wav or .m4a instead of playing")
    p.add_argument("--print", dest="dry_run", action="store_true", help="print the spoken text and exit")
    p.add_argument("--list-voices", action="store_true", help="list the model's voices and exit")
    p.add_argument("--list-models", action="store_true", help="list the model shortcuts and exit")
    p.add_argument("-q", "--quiet", action="store_true", help="don't print text while reading")
    p.add_argument("--verbose", action="store_true", help="show model output and warnings")
    args = p.parse_args(argv)

    if args.list_models:
        from .engines import MODELS

        for name, spec in MODELS.items():
            print(f"{name:<11} {spec['repo']}\n{'':<11} {spec['about']}")
        print(f"{'say':<11} macOS built-in voices")
        return

    if not 0.5 <= args.speed <= 2.0:
        p.error("--speed must be between 0.5 and 2.0")

    extra = {}
    if args.instruct:
        extra["instruct"] = args.instruct
    if args.ref_audio:
        extra["ref_audio"] = str(Path(args.ref_audio).expanduser())

    if args.list_voices:
        engine = _load(args, extra)
        for voice in engine.voices():
            print(voice)
        return

    if not args.file:
        p.error("a markdown file is required (or - for stdin)")
    text = sys.stdin.read() if args.file == "-" else _read(Path(args.file), p)

    from .speech import segments, split_long

    segs = segments(text, code=args.code)
    start = 0
    if args.start:
        start = next((i for i, s in enumerate(segs) if s.heading and args.start.lower() in s.text.lower()), -1)
        if start < 0:
            p.error(f"no heading contains {args.start!r}")
    segs = segs[start:]
    if not segs:
        p.error("nothing to read")

    if args.dry_run:
        for s in segs:
            print(f"\n{s.text}" if s.heading else s.text)
        return

    engine = _load(args, extra)
    segs = split_long(segs, engine.max_chars)

    if args.output:
        _write(engine, segs, args.output)
        return

    from .player import Session

    Session(engine, segs, 0, show_text=not args.quiet).run()


def _load(args, extra):
    from .engines import load

    print(f"\033[2mloading {args.model}…\033[0m", file=sys.stderr, flush=True)
    return load(args.model, args.voice, args.speed, extra, args.verbose)


def _read(path: Path, p: argparse.ArgumentParser) -> str:
    try:
        return path.expanduser().read_text(encoding="utf-8")
    except OSError as e:
        p.error(f"can't read {path}: {e.strerror}")


def _read_config() -> dict:
    if not CONFIG.exists():
        return {}
    try:
        with CONFIG.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"read2me: {CONFIG}: {e}")


def _write(engine, segs, output: Path) -> None:
    import numpy as np
    from scipy.io import wavfile

    from .player import render

    audio = render(engine, segs)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    output = output.expanduser()
    if output.suffix.lower() == ".wav":
        wavfile.write(output, engine.sample_rate, pcm)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            wavfile.write(tmp.name, engine.sample_rate, pcm)
            fmt = {".m4a": "m4af", ".aac": "adts", ".caf": "caff", ".aiff": "AIFF"}.get(output.suffix.lower())
            if not fmt:
                sys.exit("read2me: --output must end in .wav, .m4a, .aac, .caf or .aiff")
            data = "aac" if fmt in ("m4af", "adts") else "BEI16"
            subprocess.run(["afconvert", "-f", fmt, "-d", data, tmp.name, str(output)], check=True)
    minutes, seconds = divmod(int(len(audio) / engine.sample_rate), 60)
    print(f"wrote {output} ({minutes}:{seconds:02d})", file=sys.stderr)


if __name__ == "__main__":
    main()
