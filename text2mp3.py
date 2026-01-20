#!/usr/bin/env python3
"""
text2mp3_piper.py — Cross-platform, offline TTS (Piper) with Python-only deps
- Reads .txt or .pdf (optional page range)
- Chunks text for speed/reliability with better boundaries + punctuation
- Synthesizes WAV chunks with Piper (neural, offline)
- Inserts intentional silence between chunks (short) and paragraphs (long)
- Concatenates and encodes final MP3 using imageio-ffmpeg (bundled ffmpeg)

Usage examples:
  python text2mp3_piper.py --input chapter1.txt --model models/en_US-amy-medium.onnx --output chapter1.mp3
  python text2mp3_piper.py --input Who_Goes_There.pdf --start-page 1 --end-page 5 \
         --model models/en_US-amy-medium.onnx --output chapter1.mp3
"""

# === REQUIREMENTS (print once so you know exactly what to install) ===
print("""
Requirements to install (Python-only):
    pip install piper-tts onnxruntime imageio-ffmpeg pypdf
You'll also need to DOWNLOAD a Piper voice model (.onnx), e.g.:
    models/en_US-amy-medium.onnx
""")

# === IMPORTS (top only) ===
import argparse
import re
import sys
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
import imageio_ffmpeg


# ----------------- Text helpers -----------------

def read_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_text_from_pdf(path: Path, start_page: int | None, end_page: int | None) -> str:
    reader = PdfReader(str(path))
    n = len(reader.pages)
    s = max(1, start_page) if start_page else 1
    e = min(end_page if end_page else n, n)
    if s > e:
        raise ValueError(f"Bad page range {s}-{e}; PDF has {n} pages.")
    parts = []
    for i in range(s - 1, e):
        parts.append(reader.pages[i].extract_text() or "")
    return "\n".join(parts)


def normalize_whitespace(t: str) -> str:
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def normalize_headings(t: str) -> str:
    """
    Light heuristic: if a short all-caps line appears, treat it like a heading
    so the TTS pauses appropriately.
    """
    lines = t.splitlines()
    out = []
    for line in lines:
        s = line.strip()
        if s and len(s) < 80 and s.isupper():
            out.append(s.title() + ":")
        else:
            out.append(line)
    return "\n".join(out)


ABBREV = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "e.g.", "i.e.", "vs.", "etc.", "u.s.", "u.k.", "st.", "mt.",
}


def smart_sentence_split(p: str) -> list[str]:
    """
    Pragmatic sentence splitter that avoids splitting after common abbreviations,
    initials, and obvious numeric edge cases.
    """
    parts = re.split(r"(?<=[.!?])\s+", p.strip())
    out: list[str] = []
    buf = ""
    for s in parts:
        if not s:
            continue
        candidate = (buf + " " + s).strip() if buf else s.strip()

        last_token = candidate.split()[-1].lower()

        # Avoid splitting after common abbreviations.
        if last_token in ABBREV:
            buf = candidate
            continue

        # Avoid splitting after a single-letter initial "A."
        if re.search(r"\b[A-Z]\.$", candidate):
            buf = candidate
            continue

        # Avoid splitting after a bare integer + dot "3." (often numbered lists / decimals fragment)
        if re.search(r"\b\d+\.$", candidate):
            buf = candidate
            continue

        out.append(candidate)
        buf = ""

    if buf:
        out.append(buf)

    return out


def ensure_terminal_punct(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    # Ends with sentence punctuation (optionally followed by a quote/bracket)
    if re.search(r'[.!?]["\')\]]?\s*$', s):
        return s
    return s + "."


@dataclass
class Chunk:
    text: str
    pause_ms: int  # silence to add AFTER this chunk


def chunk_text(t: str, max_chars: int = 3000) -> list[Chunk]:
    """
    Produce chunks <= max_chars, preferring paragraph boundaries.
    Adds terminal punctuation for more natural prosody.
    Assigns pause lengths so paragraph ends get longer pauses than intra-paragraph splits.
    """
    t = t.strip()
    if len(t) <= max_chars:
        return [Chunk(ensure_terminal_punct(t), 600)]

    paras = [p.strip() for p in t.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_len = 0

    def flush(pause_ms: int):
        nonlocal cur, cur_len
        if cur:
            txt = ensure_terminal_punct("\n\n".join(cur))
            chunks.append(Chunk(txt, pause_ms))
            cur = []
            cur_len = 0

    for p in paras:
        if cur_len + len(p) + 2 <= max_chars:
            cur.append(p)
            cur_len += len(p) + 2
            continue

        # We are about to start a new chunk: treat this as a paragraph boundary pause.
        flush(pause_ms=700)

        if len(p) > max_chars:
            # Split large paragraph on smarter sentence boundaries.
            sentences = smart_sentence_split(p)
            small: list[str] = []
            s_len = 0

            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if s_len + len(s) + 1 <= max_chars:
                    small.append(s)
                    s_len += len(s) + 1
                else:
                    if small:
                        chunks.append(Chunk(ensure_terminal_punct(" ".join(small)), 220))
                    small = [s]
                    s_len = len(s)

            if small:
                chunks.append(Chunk(ensure_terminal_punct(" ".join(small)), 700))
        else:
            cur = [p]
            cur_len = len(p)

    flush(pause_ms=700)
    return chunks


# ----------------- Piper TTS via CLI -----------------

def synthesize_chunk_with_piper(
    text: str,
    model_path: Path,
    wav_out: Path,
    speaker: int | None = None,
    length_scale: float = 1.0,
    noise_scale: float = 0.667,
    noise_w: float = 0.8
):
    """
    Use Piper CLI (python -m piper) to synthesize `text` to a WAV file.
    """
    cmd = [
        sys.executable, "-m", "piper",
        "-m", str(model_path),
        "-f", str(wav_out),
        "--noise_scale", str(noise_scale),
        "--length_scale", str(length_scale),
        "--noise_w", str(noise_w),
    ]
    if speaker is not None:
        cmd += ["-s", str(speaker)]

    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


# ----------------- Concatenation & MP3 encoding -----------------

def ffmpeg_path() -> str:
    """Use imageio-ffmpeg’s bundled ffmpeg binary (cross-platform, Python-only)."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def make_silence_wav(out_path: Path, duration_ms: int, sr: int = 22050, ch: int = 1):
    """
    Create a PCM WAV of silence using ffmpeg's lavfi anullsrc.
    """
    ff = ffmpeg_path()
    duration_s = max(0.0, duration_ms / 1000.0)
    cl = "mono" if ch == 1 else "stereo"
    cmd = [
        ff, "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sr}:cl={cl}",
        "-t", f"{duration_s:.3f}",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def concat_to_wav_then_mp3(
    wav_and_pauses: list[tuple[Path, int]],
    mp3_out: Path,
    bitrate: str = "192k",
    sr: int = 22050,
    ch: int = 1,
):
    """
    Concatenate WAVs with intentional silence between them (based on pause_ms),
    then encode to MP3 using imageio-ffmpeg's ffmpeg binary.
    """
    ff = ffmpeg_path()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # 1) Uniformize each part: sr, ch, 16-bit PCM, with a tiny fade-in to reduce "cold open"
        uniform_dir = td_path / "uniform"
        uniform_dir.mkdir(exist_ok=True)

        uniform_items: list[tuple[Path, int]] = []
        for p, pause_ms in wav_and_pauses:
            u = uniform_dir / (p.stem + "_u.wav")
            cmd_u = [
                ff, "-y",
                "-i", str(p),
                "-ar", str(sr), "-ac", str(ch),
                "-af", "afade=t=in:st=0:d=0.02",
                "-c:a", "pcm_s16le",
                str(u),
            ]
            subprocess.run(cmd_u, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            uniform_items.append((u, pause_ms))

        # 2) Build concat list interleaving silence WAVs.
        list_file = td_path / "list.txt"
        lines: list[str] = []
        silence_cache: dict[int, Path] = {}

        for i, (u_wav, pause_ms) in enumerate(uniform_items):
            lines.append(f"file '{u_wav.as_posix()}'")

            # Add silence after this chunk unless it's the last one
            if i < len(uniform_items) - 1:
                pause_ms = max(0, int(pause_ms))
                if pause_ms > 0:
                    sil = silence_cache.get(pause_ms)
                    if sil is None:
                        sil = td_path / f"silence_{pause_ms}ms.wav"
                        make_silence_wav(sil, pause_ms, sr=sr, ch=ch)
                        silence_cache[pause_ms] = sil
                    lines.append(f"file '{sil.as_posix()}'")

        list_file.write_text("\n".join(lines), encoding="utf-8")

        # 3) Concat to a single WAV (safe because all PCM params match now).
        concat_wav = td_path / "concat.wav"
        cmd_concat = [
            ff, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(concat_wav),
        ]
        subprocess.run(cmd_concat, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 4) Encode to MP3
        mp3_out.parent.mkdir(parents=True, exist_ok=True)
        cmd_mp3 = [ff, "-y", "-i", str(concat_wav), "-b:a", bitrate, str(mp3_out)]
        subprocess.run(cmd_mp3, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(description="Offline Piper TTS to MP3 (chunked, cross-platform, Python-only deps)")
    ap.add_argument("--input", "-i", required=True, help="Path to .txt or .pdf")
    ap.add_argument("--output", "-o", required=True, help="Output MP3 path")
    ap.add_argument("--model", "-m", required=True, help="Path to Piper model .onnx (e.g., models/en_US-amy-medium.onnx)")
    ap.add_argument("--start-page", type=int, default=None, help="(PDF) 1-based start page")
    ap.add_argument("--end-page", type=int, default=None, help="(PDF) 1-based end page (inclusive)")
    ap.add_argument("--max-chars", type=int, default=3000, help="Chunk size for synthesis")
    # Voice controls (depend on model):
    ap.add_argument("--speaker", type=int, default=None, help="Speaker ID (for multi-speaker models)")
    ap.add_argument("--length-scale", type=float, default=1.0, help=">1.0 = slower; <1.0 = faster")
    ap.add_argument("--noise-scale", type=float, default=0.667, help="Prosody variation (0.3–1.0 typical)")
    ap.add_argument("--noise-w", type=float, default=0.8, help="Randomness (0.3–1.0 typical)")
    ap.add_argument("--bitrate", default="192k", help="MP3 bitrate (e.g., 128k, 192k, 256k)")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        print(f"Piper model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Load text
    if in_path.suffix.lower() == ".pdf":
        text = read_text_from_pdf(in_path, args.start_page, args.end_page)
    elif in_path.suffix.lower() == ".txt":
        text = read_text_from_txt(in_path)
    else:
        print("Unsupported input type. Use .pdf or .txt", file=sys.stderr)
        sys.exit(2)

    text = normalize_whitespace(text)
    text = normalize_headings(text)

    if not text:
        print("No text extracted. If the PDF is scanned, run OCR (e.g., ocrmypdf) first.", file=sys.stderr)
        sys.exit(3)

    chunks = chunk_text(text, max_chars=args.max_chars)
    print(f"[piper-tts] Synthesizing {len(chunks)} chunk(s) with model: {model_path.name}")

    # Generate WAV parts
    wav_and_pauses: list[tuple[Path, int]] = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for i, ch in enumerate(chunks, start=1):
            wav_out = td_path / f"part_{i:03d}.wav"
            synthesize_chunk_with_piper(
                ch.text,
                model_path,
                wav_out,
                speaker=args.speaker,
                length_scale=args.length_scale,
                noise_scale=args.noise_scale,
                noise_w=args.noise_w,
            )
            wav_and_pauses.append((wav_out, ch.pause_ms))

        # Concatenate and encode MP3
        out_mp3 = Path(args.output).expanduser().resolve()
        concat_to_wav_then_mp3(wav_and_pauses, out_mp3, bitrate=args.bitrate)

    print(f"[piper-tts] Done → {out_mp3}")


if __name__ == "__main__":
    main()