# text2mp3 (Piper TTS)

A **cross-platform, fully offline text-to-MP3 pipeline** built on **Piper TTS**.  
Designed for long-form reading (ebooks, essays, Project Gutenberg texts) with **natural pacing**, **clean chunk boundaries**, and **intentional pauses** that sound like real speech rather than stitched audio.

This tool:
- Reads `.txt` or `.pdf` (optional page ranges)
- Cleans and normalizes text for speech
- Chunks intelligently (paragraph-aware, sentence-safe)
- Inserts natural pauses between chunks and paragraphs
- Synthesizes speech using **Piper** (offline neural TTS)
- Concatenates and encodes a final MP3 using a bundled `ffmpeg`

No cloud APIs. No external services. Just Python.

---

## Why Piper?

Piper provides **high-quality neural TTS** that runs **entirely offline** and scales well to book-length content.  
Among the English voices, **Alan (en_GB-alan-medium)** stands out for long narrative reads — steady cadence, clear diction, and minimal fatigue over hours of listening.

**Recommended voice:** `en_GB-alan-medium.onnx`

---

## Requirements

Python 3.10+

Install dependencies:

```bash
pip install piper-tts onnxruntime imageio-ffmpeg pypdf
```

Download a Piper voice model (example):

```text
models/en_GB-alan-medium.onnx
```

Voice models are available from:  
https://github.com/rhasspy/piper

---

## Basic Usage

### Project Gutenberg (plain text)

Works especially well with Project Gutenberg texts.

**Example:**

```bash
python3 text2mp3.py \
  -i The_Man_Eaters_of_Tsavo.txt \
  -m models/en_GB-alan-medium.onnx \
  -o The_Man_Eaters_of_Tsavo_alan_Piper.mp3
```

---

## PDF Input

```bash
python3 text2mp3.py \
  -i Who_Goes_There.pdf \
  --start-page 1 \
  --end-page 5 \
  -m models/en_GB-alan-medium.onnx \
  -o Who_Goes_There_excerpt.mp3
```

> If your PDF is scanned, run OCR first (e.g. `ocrmypdf`).

---

## Key Options

| Option | Description |
|------|------------|
| `--max-chars` | Chunk size (default: 3000). Larger = fewer joins |
| `--length-scale` | Speech speed (`>1.0` slower, `<1.0` faster) |
| `--noise-scale` | Prosody variation |
| `--noise-w` | Randomness / expressiveness |
| `--bitrate` | MP3 bitrate |
| `--speaker` | Speaker ID (multi-speaker models) |

---

## Recommended Settings (Audiobook Style)

These are now the tool's defaults (as of the Cori-high tuning pass), so you
only need to pass them explicitly if you want something different:

```bash
--max-chars 600 \
--length-scale 1.20 \
--noise-scale 0.6 \
--noise-w 0.75 \
--volume 1.4 \
--bitrate 160k
```

> 160k is the ceiling for MP3 at these voices' 22.05kHz sample rate (the
> MPEG-2 LSF format has no higher option below 32kHz) -- `text2mp3.py`
> clamps anything higher to 160k automatically, so requesting more doesn't
> do anything.
>
> `--max-chars 600` trades a few more chunk splices (invisible now that
> edges are fade-smoothed) for a real, measured reduction in the rate of
> the occasional dropped/slurred word neural TTS models are prone to on
> longer input sequences -- a known general limitation of VITS-style
> models, not something this pipeline can fully eliminate.
>
> `--volume 1.4` exists because `normalize_audio` is off by default (it
> was flattening natural loudness variation across chunks); this brings
> the level back up without reintroducing that flattening.

---

## Notes

- Fully offline
- Stable for multi-hour books
- No GPU required

---

## Credits

- Piper TTS — https://github.com/rhasspy/piper  
- Project Gutenberg — https://www.gutenberg.org/

---

**Piper Alan + Project Gutenberg = excellent long-form listening.**
