"""Turn a markdown document into a list of segments a TTS model can read well.

Speech models read what they are given. A path like `src/proxy/auth.ts`, a label like
`[High]`, a date like 2026-11-04 or a table pipe gets read literally or garbled, so this
module rewrites the document into plain spoken English first.
"""

from __future__ import annotations

import calendar
import re
import unicodedata
from dataclasses import dataclass

from markdown_it import MarkdownIt
from markdown_it.token import Token


@dataclass
class Segment:
    text: str
    pause: float = 0.3  # seconds of silence after this segment
    heading: int = 0  # heading level, 0 when not a heading


# Silence, in seconds, after each kind of block.
PAUSE_HEADING = 0.45
PAUSE_BEFORE_HEADING = 0.7
PAUSE_PARAGRAPH = 0.35
PAUSE_ITEM = 0.2
PAUSE_SENTENCE = 0.05

FRONTMATTER = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*(\n|\Z)", re.S)
URL = re.compile(r"\bhttps?://[^\s)>\]]+|\bwww\.[^\s)>\]]+")
SEMVER = re.compile(r"\b(v)?(\d+)\.(\d+)\.(\d+)\b")
ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
ISSUE_RANGE = re.compile(r"(\b(?:PRs?|issues?|tickets?|requests?)\s+)?#(\d+)\s*[–—-]\s*#?(\d+)\b", re.I)
ISSUE = re.compile(r"(\b(?:PRs?|issues?|tickets?|requests?)\s+)?(?<![\w&])#(\d+)\b", re.I)
TICKET = re.compile(r"\b([A-Z][A-Z0-9]+)-(\d+)\b")
NUMBER_RANGE = re.compile(r"(?<=\d)\s*[–]\s*(?=\d)")
RISK = re.compile(r"^\[[^\]]+\]\s*(?:→|->|=>)")
COMMAND = re.compile(r"\s\|\s|&&|[;'\"$*{}]")
CALLOUT = re.compile(r"^\[!(\w+)\][+-]?\s*", re.I)
TASK = re.compile(r"^\[([ xX])\]\s+")
CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z0-9])(?=[A-Z][a-z])")
SHOUTED = re.compile(r"\b[A-Z]{5,}\b")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

PROSE = [
    (re.compile(r"\be\.g\.,?", re.I), "for example,"),
    (re.compile(r"\bi\.e\.,?", re.I), "that is,"),
    (re.compile(r"\betc\.(?=\s+[a-z])"), "etcetera"),
    (re.compile(r"\betc\."), "etcetera."),
    (re.compile(r"\bvs\.?(?=\s)"), "versus"),
    (re.compile(r"\bw/(?=\s)"), "with"),
    (re.compile(r"\s*(?:->|→|⟶|=>|⇒)\s*"), " to "),
    (re.compile(r"\s*(?:<-|←)\s*"), " from "),
    (re.compile(r"\s*(?:<->|↔)\s*"), " and "),
    (re.compile(r"\s*≥\s*"), " at least "),
    (re.compile(r"\s*≤\s*"), " at most "),
    (re.compile(r"\s*≠\s*"), " is not "),
    (re.compile(r"\s*[≈~](?=\d)"), " about "),
    (re.compile(r"\s+&\s+"), " and "),
    (re.compile(r"\s*—\s*|\s+–\s+|\s+--\s+"), ", "),
    (re.compile(r"(?<=\w)\(s\)"), "s"),
]

LANGUAGES = {
    "ts": "TypeScript", "typescript": "TypeScript", "tsx": "TypeScript",
    "js": "JavaScript", "javascript": "JavaScript", "jsx": "JavaScript",
    "py": "Python", "python": "Python", "rb": "Ruby", "go": "Go", "rs": "Rust",
    "sh": "shell", "bash": "shell", "zsh": "shell", "shell": "shell", "console": "shell",
    "json": "JSON", "jsonc": "JSON", "yaml": "YAML", "yml": "YAML", "toml": "TOML",
    "sql": "SQL", "html": "HTML", "xml": "XML", "css": "CSS", "diff": "diff",
    "text": "", "txt": "", "plaintext": "", "md": "markdown", "markdown": "markdown",
}


def segments(markdown: str, code: str = "announce") -> list[Segment]:
    """Parse markdown into spoken segments. `code` is announce, read or skip."""
    md = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    tokens = md.parse(FRONTMATTER.sub("", markdown, count=1))
    return _Walker(code).walk(tokens)


def split_long(segs: list[Segment], max_chars: int) -> list[Segment]:
    """Split segments longer than max_chars at sentence boundaries."""
    out: list[Segment] = []
    for seg in segs:
        if len(seg.text) <= max_chars:
            out.append(seg)
            continue
        pieces: list[str] = []
        for sentence in SENTENCE_END.split(seg.text):
            if pieces and len(pieces[-1]) + len(sentence) + 1 <= max_chars:
                pieces[-1] += " " + sentence
            else:
                pieces.append(sentence)
        for piece in pieces[:-1]:
            out.append(Segment(piece, PAUSE_SENTENCE, seg.heading))
        out.append(Segment(pieces[-1], seg.pause, seg.heading))
    return out


class _Walker:
    def __init__(self, code: str):
        self.code = code
        self.out: list[Segment] = []
        self.lists: list[list] = []  # [ordered, counter] per open list
        self.prefix = ""  # spoken before the next paragraph, e.g. a list number

    def walk(self, tokens: list[Token]) -> list[Segment]:
        i = 0
        while i < len(tokens):
            t = tokens[i]
            kind = t.type
            if kind == "heading_open":
                self._pause_before(PAUSE_BEFORE_HEADING)
                self._emit(inline(tokens[i + 1]), PAUSE_HEADING, heading=int(t.tag[1]))
                i += 3
                continue
            if kind == "paragraph_open":
                self._paragraph(inline(tokens[i + 1]), tokens[i + 1].content)
                i += 3
                continue
            if kind in ("bullet_list_open", "ordered_list_open"):
                start = int(t.attrs.get("start", 1) or 1) if kind == "ordered_list_open" else 1
                self.lists.append([kind == "ordered_list_open", start - 1])
            elif kind in ("bullet_list_close", "ordered_list_close"):
                self.lists.pop()
                self._pause_before(PAUSE_PARAGRAPH)
            elif kind == "list_item_open":
                ordered = self.lists[-1]
                if ordered[0]:
                    ordered[1] += 1
                    self.prefix = f"{ordered[1]}. "
            elif kind in ("fence", "code_block"):
                self._code(t)
            elif kind == "table_open":
                i = self._table(tokens, i)
                continue
            elif kind == "hr":
                self._pause_before(PAUSE_BEFORE_HEADING)
            elif kind == "html_block":
                text = strip_html(t.content)
                if text:
                    self._emit(prose(text), PAUSE_PARAGRAPH)
            i += 1
        return [s for s in self.out if s.text.strip()]

    def _paragraph(self, text: str, raw: str) -> None:
        if RISK.match(raw):
            # OpenSpec's design.md convention: "- [Risk] → Mitigation"
            risk, _, mitigation = text.partition("] to ")
            text = f"Risk: {sentence(risk.lstrip('['))} Mitigation: {mitigation}"
        callout = CALLOUT.match(text)
        if callout:
            text = f"{callout.group(1).capitalize()}: {text[callout.end():]}"
        task = TASK.match(text)
        if task and self.lists:
            text = ("Done: " if task.group(1).lower() == "x" else "") + text[task.end():]
        pause = PAUSE_ITEM if self.lists else PAUSE_PARAGRAPH
        self._emit(self.prefix + text, pause)
        self.prefix = ""

    def _code(self, t: Token) -> None:
        if self.code == "skip":
            return
        lang = (t.info or "").split()[0].lower() if t.info else ""
        lines = [line for line in t.content.splitlines() if line.strip()]
        if self.code == "read":
            for line in lines:
                self._emit(speak_code(line), PAUSE_ITEM)
            return
        if lang == "mermaid":
            self._emit("Skipping a diagram.", PAUSE_PARAGRAPH)
            return
        name = LANGUAGES.get(lang, lang)
        size = f"{len(lines)}-line " if len(lines) > 1 else ""
        self._emit(f"Skipping a {size}{name + ' ' if name else ''}code block.", PAUSE_PARAGRAPH)

    def _table(self, tokens: list[Token], i: int) -> int:
        head: list[str] = []
        rows: list[list[str]] = []
        row: list[str] | None = None
        in_head = False
        while tokens[i].type != "table_close":
            t = tokens[i]
            if t.type == "thead_open":
                in_head = True
            elif t.type == "thead_close":
                in_head = False
            elif t.type == "tr_open":
                row = []
            elif t.type == "tr_close" and row is not None:
                (head.extend(row) if in_head else rows.append(row))
            elif t.type == "inline" and row is not None:
                row.append(inline(t))
            i += 1
        self._emit(f"A table with {len(rows)} row{'s' if len(rows) != 1 else ''}.", PAUSE_ITEM)
        for cells in rows:
            parts = []
            for n, cell in enumerate(cells):
                if not cell:
                    continue
                label = head[n] if n < len(head) else ""
                parts.append(f"{label}: {cell}" if label and len(cells) > 1 else cell)
            self._emit(". ".join(sentence(p).rstrip(".") for p in parts), PAUSE_ITEM)
        self._pause_before(PAUSE_PARAGRAPH)
        return i + 1

    def _emit(self, text: str, pause: float, heading: int = 0) -> None:
        # Brackets left in the text are unmatched link syntax, never spoken
        text = sentence(re.sub(r"[\[\]]", "", text))
        if text:
            self.out.append(Segment(text, pause, heading))

    def _pause_before(self, seconds: float) -> None:
        if self.out:
            self.out[-1].pause = max(self.out[-1].pause, seconds)


def inline(tok: Token) -> str:
    """Render one inline token (a paragraph, heading or cell) as spoken text."""
    parts: list[str] = []
    link_start = 0
    href = ""
    for c in tok.children or []:
        kind = c.type
        if kind == "text":
            parts.append(prose(c.content))
        elif kind == "code_inline":
            parts.append(speak_code(c.content))
        elif kind in ("softbreak", "hardbreak"):
            parts.append(" ")
        elif kind == "link_open":
            href = str(c.attrs.get("href", ""))
            link_start = len(parts)
        elif kind == "link_close":
            label = "".join(parts[link_start:]).strip()
            if not label or URL.fullmatch(label) or label == href or label.startswith("a link to"):
                parts[link_start:] = [speak_url(href)]
        elif kind == "image":
            alt = c.content.strip()
            parts.append(f"(image: {prose(alt)})" if alt else "")
        elif kind == "html_inline":
            if re.match(r"<br\s*/?>", c.content, re.I):
                parts.append(" ")
    return collapse("".join(parts))


def prose(text: str) -> str:
    """Normalize ordinary text: abbreviations, symbols, dates, issue numbers, URLs."""
    text = URL.sub(lambda m: speak_url(m.group(0)), text)
    # "PR #21" reads as "PR 21"; a bare "#21" needs the word "number"
    text = ISSUE_RANGE.sub(lambda m: f"{m.group(1) or 'numbers '}{m.group(2)} to {m.group(3)}", text)
    text = ISSUE.sub(lambda m: f"{m.group(1) or 'number '}{m.group(2)}", text)
    text = ISO_DATE.sub(_speak_date, text)
    text = SEMVER.sub(_speak_version, text)
    text = TICKET.sub(r"\1 \2", text)
    text = NUMBER_RANGE.sub(" to ", text)
    for pattern, spoken in PROSE:
        text = pattern.sub(spoken, text)
    return strip_symbols(text)


def speak_code(code: str) -> str:
    """Say an inline code span: paths, identifiers, labels and versions."""
    s = code.strip()
    if URL.fullmatch(s):
        return speak_url(s)
    s = re.sub(r"<!--.*?-->", "HTML comment", s)
    if " " in s and COMMAND.search(s) and len(s) > 20:
        return "a code snippet"  # a shell command or expression; reading it is noise
    s = s.lstrip("#").strip().rstrip("/")
    if re.fullmatch(r"\[[^\]]+\]", s):  # a label like [High]
        s = s[1:-1]
    s = re.sub(r"\[(\w+)\]", r" \1", s)  # claude[bot]
    s = SEMVER.sub(_speak_version, s)
    s = ISO_DATE.sub(_speak_date, s)
    s = TICKET.sub(r"\1 \2", s)
    s = re.sub(r"(?<=\w)\.(?=[A-Za-z])", " dot ", s)  # REVIEW.md, obj.method
    s = re.sub(r"(?:(?<=^)|(?<=[\s/]))\.(?=\w)", "dot ", s)  # .github
    s = s.replace("()", "")
    s = re.sub(r"\s*/\s*", " slash ", s).strip()
    s = s.removeprefix("slash ")
    s = s.replace("@", " at ").replace("=", " equals ").replace("&&", " and ")
    s = re.sub(r"[_]+|(?<=[A-Za-z0-9])-(?=[A-Za-z])", " ", s)
    s = CAMEL.sub(" ", s)
    s = SHOUTED.sub(lambda m: m.group(0).capitalize(), s)
    s = re.sub(r"[<>{}`$*|\\^]", " ", s)
    return collapse(s)


def speak_url(url: str) -> str:
    host = re.sub(r"^(https?://)?(www\.)?", "", url).split("/")[0]
    return f"a link to {host}" if host else "a link"


def strip_html(html: str) -> str:
    html = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    return collapse(re.sub(r"<[^>]+>", " ", html))


def strip_symbols(text: str) -> str:
    """Drop emoji and pictographs, which models either skip or read as noise."""
    return "".join(
        ch for ch in text if unicodedata.category(ch) not in ("So", "Cs") and ch != "️"
    )


def sentence(text: str) -> str:
    """Trim and make sure the text ends like a sentence, so the model lands it."""
    text = collapse(text).strip(" ,;")
    if text and text[-1] not in ".!?:\"')":
        text += "."
    return text


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _speak_date(m: re.Match) -> str:
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return m.group(0)
    return f"{calendar.month_name[month]} {day}, {year}"


def _speak_version(m: re.Match) -> str:
    spoken = " point ".join(m.group(i) for i in (2, 3, 4))
    return f"version {spoken}" if m.group(1) else spoken
