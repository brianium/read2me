import sys
import threading
import types

import numpy as np

from read2me.engines import Engine
from read2me.player import Session
from read2me.speech import Segment

RATE = 1000


class FakeEngine(Engine):
    sample_rate = RATE

    def __init__(self):
        self.calls = []

    def synth(self, text):
        self.calls.append(text)
        # encode the segment number in the audio so playback order is checkable
        return np.full(int(0.25 * RATE), float(text.split()[-1]) / 100, np.float32)


class FakeStream:
    def __init__(self, written, on_write=None, **_):
        self.written = written
        self.on_write = on_write

    def start(self):
        pass

    def write(self, block):
        self.written.append(np.asarray(block).copy())
        if self.on_write:
            self.on_write()

    def stop(self):
        pass

    def close(self):
        pass


def fake_sounddevice(monkeypatch, written, on_write=None):
    module = types.SimpleNamespace(OutputStream=lambda **kw: FakeStream(written, on_write, **kw))
    monkeypatch.setitem(sys.modules, "sounddevice", module)


def played(written):
    """Segment numbers in the order their audio reached the device."""
    order = []
    for value in np.concatenate(written):
        n = round(float(value) * 100)
        if n and (not order or order[-1] != n):
            order.append(n)
    return order


def segs(n):
    return [Segment(f"segment {i}", pause=0.0) for i in range(1, n + 1)]


def run(session):
    t = threading.Thread(target=session.run)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "session hung"


def test_plays_every_segment_in_order(monkeypatch):
    written = []
    fake_sounddevice(monkeypatch, written)
    engine = FakeEngine()
    run(Session(engine, segs(5), 0, show_text=False))
    assert played(written) == [1, 2, 3, 4, 5]
    assert engine.calls == [f"segment {i}" for i in range(1, 6)]


def test_seek_skips_ahead(monkeypatch):
    written = []
    session = None

    def jump_once():
        if len(written) == 1:
            session.seek(3)  # from segment 1 straight to segment 4

    fake_sounddevice(monkeypatch, written, jump_once)
    session = Session(FakeEngine(), segs(5), 0, show_text=False)
    run(session)
    assert played(written) == [1, 4, 5]


def test_section_navigation(monkeypatch):
    fake_sounddevice(monkeypatch, [])
    items = [Segment("A.", heading=2), Segment("x 1"), Segment("B.", heading=2), Segment("y 2")]
    session = Session(FakeEngine(), items, 0, show_text=False)
    session.pos = 1
    session.seek_section(+1)
    assert session.jump == 2
    session.pos = 3
    session.seek_section(-1)
    assert session.jump == 2


def test_quit_stops_generation_and_playback(monkeypatch):
    written = []
    session = None

    def quit_early():
        if len(written) == 2:
            session.stop()

    fake_sounddevice(monkeypatch, written, quit_early)
    engine = FakeEngine()
    session = Session(engine, segs(200), 0, show_text=False)
    run(session)
    assert played(written)[0] == 1
    assert len(engine.calls) < 200


def test_text_shows_while_engine_hides_its_own_output(monkeypatch):
    """The engine swaps sys.stdout out while generating; the player's text must still print."""
    import contextlib
    import io

    class ChattyEngine(FakeEngine):
        def synth(self, text):
            with contextlib.redirect_stdout(io.StringIO()):
                print("model chatter")
                return super().synth(text)

    terminal = io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", terminal)
    fake_sounddevice(monkeypatch, [])
    run(Session(ChattyEngine(), segs(3), 0, show_text=True))
    shown = terminal.getvalue()
    assert all(f"segment {i}" in shown for i in (1, 2, 3))
    assert "model chatter" not in shown
