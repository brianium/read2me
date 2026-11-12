# read2me

Read a markdown file aloud with a natural-sounding voice that runs entirely on your Mac.
Free, private, and offline after the first run.

```sh
read2me design.md
```

## Requirements

- A Mac with Apple Silicon (M1 or later), macOS 14 or later
- About 1 GB of disk: 0.7 GB for the tool, 0.4 GB for the voice model

## Install

```sh
brew install uv
uv tool install git+https://github.com/brianium/read2me
```

That's it. uv fetches the right Python version itself, and nothing else needs installing.
The first `read2me` run downloads the voice model (about 400 MB).

Upgrade with `uv tool upgrade read2me`. Uninstall with `uv tool uninstall read2me`.

## Use

```sh
read2me design.md                          # read it aloud
read2me design.md --from Decisions         # start at a heading
read2me design.md -v bm_george -s 1.15     # another voice, a little faster
read2me design.md -v af_heart,af_bella     # blend two voices
read2me design.md -o design.m4a            # save audio instead of playing
read2me design.md --print                  # see exactly what will be spoken
read2me --list-voices
cat notes.md | read2me -
```

While it plays:

| Key | Action |
|---|---|
| space | pause / resume |
| ← → | previous / next sentence |
| ↑ ↓ | previous / next section |
| + − | faster / slower |
| q | quit |

To change your defaults, create `~/.config/read2me/config.toml`:

```toml
voice = "af_heart"
speed = 1.1
```

## Voices

The default voice is `af_heart`. `read2me --list-voices` lists all 28 English ones:
`af_*` and `am_*` are American female and male, `bf_*` and `bm_*` are British.
`af_heart` and `af_bella` are the strongest; `bm_george` and `am_michael` are good male voices.

## Other models

The default model is [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M): small, fast and
accurate. Larger models are available with `-m`. They download on first use and are slower.

| `-m` | Download | Speed | Notes |
|---|---|---|---|
| `kokoro` (default) | 0.4 GB | ~20× realtime | Most accurate in testing |
| `qwen3` | 4.5 GB | ~2× | Expressive; set a style with `--instruct "calm narration"`. Occasionally drops a word |
| `voxtral` | 8 GB | ~1.7× | Mistral's model. Licensed CC BY-NC 4.0, personal use only |
| `chatterbox` | 3 GB | ~4× | Clone a voice from a short clip with `--ref-audio clip.wav` |
| `say` | none | instant | The built-in macOS voice |

Any [mlx-audio](https://github.com/Blaizzy/mlx-audio) model repo on Hugging Face also works,
e.g. `-m mlx-community/Kokoro-82M-bf16`.

## How it reads markdown

Before the voice model sees the text, read2me rewrites it into plain spoken English:

- Headings get a pause around them; `--from` and ↑ ↓ use them to navigate.
- `src/api/auth.ts` is read as "src slash api slash auth dot ts"; `[High]` as "High".
- `2026-11-04` becomes "November 4, 2026", `v1.4.0` "version 1 point 4 point 0", and
  `PRs #21–#23` "PRs 21 to 23".
- Tables are read row by row, each cell with its column header.
- Code blocks are announced and skipped (`--code read` reads them, `--code skip` is silent).
- Links read their text; bare URLs read as "a link to github.com".
- [OpenSpec](https://github.com/Fission-AI/OpenSpec)-style `[Risk] → Mitigation` bullets are
  read as "Risk: … Mitigation: …".

`--print` shows the result without speaking it.

## Development

```sh
git clone https://github.com/brianium/read2me && cd read2me
uv run read2me README.md
uv run --group dev pytest
```

Install your working copy with `uv tool install --editable .`.

## License

MIT. The voice models have their own licenses, listed above.
