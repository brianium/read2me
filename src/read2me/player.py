"""Generate ahead of playback and play segments as they become ready.

MLX work stays on the main thread (MLX streams are per-thread), so generation runs here
and playback and the keyboard each get a background thread. Output goes to sys.__stdout__
and sys.__stderr__ because the engine swaps sys.stdout out while it generates.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import time
import tty

import numpy as np

from .engines import Engine
from .speech import Segment

LOOKAHEAD = 30  # segments generated ahead of the one playing
BLOCK_SECONDS = 0.1

KEYS = "space pause · ←/→ segment · ↑/↓ section · +/- speed · q quit"


class Session:
    def __init__(self, engine: Engine, segs: list[Segment], start: int, show_text: bool):
        self.engine = engine
        self.segs = segs
        self.audio: dict[int, np.ndarray] = {}
        self.cond = threading.Condition()
        self.pos = start  # segment being played
        self.jump: int | None = None
        self.paused = False
        self.done = False
        self.show_text = show_text

    # --- generation, main thread -------------------------------------------------

    def run(self) -> None:
        player = threading.Thread(target=self._play, daemon=True)
        keys = _Keys(self) if sys.stdin.isatty() else None
        if keys:
            keys.start()
            print(f"\033[2m{KEYS}\033[0m", file=sys.__stderr__)
        player.start()
        try:
            while not self.done:
                with self.cond:
                    i = self._next_to_generate()
                    if i is None:
                        self.cond.wait(0.2)
                        continue
                    seg = self.segs[i]
                try:
                    audio = self.engine.synth(seg.text)
                except Exception as e:  # one bad segment shouldn't end the reading
                    _status(f"skipped segment {i + 1}: {e}")
                    audio = np.zeros(0, np.float32)
                silence = np.zeros(int(seg.pause * self.engine.sample_rate), np.float32)
                with self.cond:
                    self.audio[i] = np.concatenate([audio, silence])
                    self.cond.notify_all()
        except KeyboardInterrupt:
            self.stop()
        finally:
            player.join(timeout=1)
            if keys:
                keys.restore()

    def _next_to_generate(self) -> int | None:
        end = min(len(self.segs), self.pos + LOOKAHEAD)
        return next((i for i in range(self.pos, end) if i not in self.audio), None)

    def stop(self) -> None:
        with self.cond:
            self.done = True
            self.cond.notify_all()

    # --- controls, keyboard thread -----------------------------------------------

    def seek(self, i: int) -> None:
        with self.cond:
            self.jump = max(0, min(len(self.segs) - 1, i))
            self.paused = False
            self.cond.notify_all()

    def seek_section(self, direction: int) -> None:
        here = self.pos
        if direction < 0:
            # back to the start of this section, or the previous one if already there
            heads = [i for i, s in enumerate(self.segs[:here]) if s.heading]
            self.seek(heads[-1] if heads else 0)
        else:
            nxt = next((i for i in range(here + 1, len(self.segs)) if self.segs[i].heading), None)
            if nxt is not None:
                self.seek(nxt)

    def change_speed(self, delta: float) -> None:
        if not hasattr(self.engine, "speed"):
            return
        with self.cond:
            self.engine.speed = round(min(2.0, max(0.5, self.engine.speed + delta)), 2)
            # drop audio generated at the old speed, except what is playing now
            for i in [i for i in self.audio if i > self.pos]:
                del self.audio[i]
            self.cond.notify_all()
        _status(f"speed {self.engine.speed:.2f}")

    def toggle_pause(self) -> None:
        with self.cond:
            self.paused = not self.paused
            self.cond.notify_all()
        _status("paused" if self.paused else "playing")

    # --- playback thread ---------------------------------------------------------

    def _play(self) -> None:
        import sounddevice as sd

        stream = None
        i = self.pos
        try:
            while i < len(self.segs) and not self.done:
                with self.cond:
                    while i not in self.audio and not self.done and self.jump is None:
                        self.cond.wait(0.1)
                    if self.jump is not None:
                        i, self.jump = self.jump, None
                        self.pos = i
                        continue
                    if self.done:
                        break
                    audio = self.audio[i]
                if stream is None:
                    stream = sd.OutputStream(samplerate=self.engine.sample_rate, channels=1, dtype="float32")
                    stream.start()
                self._show(i)
                i = self._play_segment(stream, audio, i)
                with self.cond:
                    self.pos = i
                    # keep a little behind for ←, free the rest
                    for old in [k for k in self.audio if k < i - 5]:
                        del self.audio[old]
                    self.cond.notify_all()
        finally:
            if stream is not None:
                if not self.done:
                    stream.write(np.zeros(int(0.3 * self.engine.sample_rate), np.float32))
                stream.stop()
                stream.close()
            self.stop()

    def _play_segment(self, stream, audio: np.ndarray, i: int) -> int:
        block = int(BLOCK_SECONDS * self.engine.sample_rate)
        k = 0
        while k < len(audio):
            if self.done:
                return i
            if self.jump is not None:
                with self.cond:
                    target, self.jump = self.jump, None
                return target
            if self.paused:
                time.sleep(0.05)
                continue
            stream.write(audio[k : k + block])
            k += block
        return i + 1

    def _show(self, i: int) -> None:
        if not self.show_text:
            return
        seg = self.segs[i]
        width = len(str(len(self.segs)))
        tag = f"\033[2m{i + 1:>{width}}/{len(self.segs)}\033[0m"
        text = f"\033[1m{seg.text}\033[0m" if seg.heading else seg.text
        print(f"{tag} {text}", file=sys.__stdout__, flush=True)


def render(engine: Engine, segs: list[Segment]) -> np.ndarray:
    """Generate the whole document without playing it."""
    parts = []
    for n, seg in enumerate(segs, 1):
        print(f"\r\033[Krendering {n}/{len(segs)}", end="", file=sys.__stderr__, flush=True)
        parts.append(engine.synth(seg.text))
        parts.append(np.zeros(int(seg.pause * engine.sample_rate), np.float32))
    print(file=sys.__stderr__)
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def _status(message: str) -> None:
    print(f"\033[2m  [{message}]\033[0m", file=sys.__stderr__, flush=True)


class _Keys(threading.Thread):
    """Read single keypresses without waiting for Enter."""

    def __init__(self, session: Session):
        super().__init__(daemon=True)
        self.session = session
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def restore(self) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def run(self) -> None:
        s = self.session
        actions = {
            " ": s.toggle_pause,
            "q": s.stop,
            "n": lambda: s.seek(s.pos + 1),
            "p": lambda: s.seek(s.pos - 1),
            "\x1b[C": lambda: s.seek(s.pos + 1),
            "\x1b[D": lambda: s.seek(s.pos - 1),
            "\x1b[B": lambda: s.seek_section(+1),
            "\x1b[A": lambda: s.seek_section(-1),
            "]": lambda: s.seek_section(+1),
            "[": lambda: s.seek_section(-1),
            "+": lambda: s.change_speed(+0.1),
            "=": lambda: s.change_speed(+0.1),
            "-": lambda: s.change_speed(-0.1),
        }
        while not s.done:
            ready, _, _ = select.select([self.fd], [], [], 0.2)
            if not ready:
                continue
            key = os.read(self.fd, 8).decode(errors="ignore")
            action = actions.get(key)
            if action:
                action()
