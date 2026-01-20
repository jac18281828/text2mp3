#!/usr/bin/env python3
"""
text2mp3.py — Offline Piper TTS with multi-process chunk synthesis (macOS ARM safe)

Features:
- TXT / PDF input
- Smart chunking with paragraph-aware pauses
- Multi-process Piper synthesis (CPU-only, Apple Silicon tuned)
- Intentional silence between chunks
- MP3 output via bundled ffmpeg (imageio-ffmpeg)

Recommended on Apple Silicon:
  OMP_NUM_THREADS=1
  --workers 6   (M3 Max 16-core)
"""

# ===================== Imports =====================

import argparse
import os
import re
import sys
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from pypdf import PdfReader
import imageio_ffmpeg


# ===================== Text helpers =====================

def read_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_text_from_pdf(path: Path, start_page: int | None, end_page: int | None) -> str:
    reader = PdfReader(str(path))
    n = len(reader.pages)
    s = max(1, start_page) if start_page else 1
    e = min(end_page if end_page else n, n)
    if s > e:
        raise ValueError(f"Bad page range {s}-{e}; PDF has {n} pages.")
    return "\n".join(reader.pages[i].extract_text() or "" for i in range(s - 1, e))


def normalize_whitespace(t: str) -> str:
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def normalize_headings(t: str) -> str:
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
    parts = re.split(r"(?<=[.!?])\s+", p.strip())
    out, buf = [], ""
    for s in parts:
        if not s:
            continue
        candidate = (buf + " " + s).strip() if buf else s.strip()
        last = candidate.split()[-1].lower()

        if last in ABBREV or re.search(r"\b[A-Z]\.$", candidate) or re.search(r"\b\d+\.$", candidate):
            buf = candidate
            continue

        out.append(candidate)
        buf = ""
    if buf:
        out.append(buf)
    return out


def ensure_terminal_punct(s: str) -> str:
    if not s:
        return s
    if re.search(r'[.!?]["\')\]]?\s*$', s):
        return s
    return s + "."


@dataclass
class Chunk:
    text: str
    pause_ms: int


def chunk_text(t: str, max_chars: int) -> list[Chunk]:
    if len(t) <= max_chars:
        return [Chunk(ensure_terminal_punct(t), 700)]

    paras = [p.strip() for p in t.split("\n\n") if p.strip()]
    chunks, cur, cur_len = [], [], 0

    def flush(pause):
        nonlocal cur, cur_len
        if cur:
            chunks.append(Chunk(ensure_terminal_punct("\n\n".join(cur)), pause))
            cur, cur_len = [], 0

    for p in paras:
        if cur_len + len(p) + 2 <= max_chars:
            cur.append(p)
            cur_len += len(p) + 2
            continue

        flush(700)

        if len(p) > max_chars:
            sents, buf, blen = smart_sentence_split(p), [], 0
            for s in sents:
                if blen + len(s) + 1 <= max_chars:
                    buf.append(s)
                    blen += len(s) + 1
                else:
                    chunks.append(Chunk(ensure_terminal_punct(" ".join(buf)), 220))
                    buf, blen = [s], len(s)
            if buf:
                chunks.append(Chunk(ensure_terminal_punct(" ".join(buf)), 700))
        else:
            cur, cur_len = [p], len(p)

    flush(700)
    return chunks


# ===================== Piper synthesis =====================

def synthesize_chunk_with_piper(
    text: str,
    model_path: Path,
    wav_out: Path,
    speaker: int | None,
    length_scale: float,
    noise_scale: float,
    noise_w: float,
):
    cmd = [
        sys.executable, "-m", "piper",
        "-m", str(model_path),
        "-f", str(wav_out),
        "--length_scale", str(length_scale),
        "--noise_scale", str(noise_scale),
        "--noise_w", str(noise_w),
    ]
    if speaker is not None:
        cmd += ["-s", str(speaker)]

    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


def _synth_worker(job):
    i, text, model, wav, speaker, ls, ns, nw = job
    synthesize_chunk_with_piper(
        text, Path(model), Path(wav),
        speaker, ls, ns, nw
    )
    return i, wav


# ===================== Audio concat =====================

def ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def make_silence_wav(out: Path, ms: int, sr=22050):
    ff = ffmpeg_path()
    subprocess.run(
        [ff, "-y", "-f", "lavfi",
         "-i", f"anullsrc=r={sr}:cl=mono",
         "-t", f"{ms/1000:.3f}",
         "-c:a", "pcm_s16le", str(out)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def concat_to_mp3(wavs_and_pauses, out_mp3: Path, bitrate="192k"):
    ff = ffmpeg_path()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lst, silence_cache = [], {}

        for i, (wav, pause) in enumerate(wavs_and_pauses):
            lst.append(f"file '{wav.as_posix()}'")
            if i < len(wavs_and_pauses) - 1 and pause > 0:
                sil = silence_cache.get(pause)
                if not sil:
                    sil = td / f"sil_{pause}.wav"
                    make_silence_wav(sil, pause)
                    silence_cache[pause] = sil
                lst.append(f"file '{sil.as_posix()}'")

        list_file = td / "list.txt"
        list_file.write_text("\n".join(lst))

        concat = td / "all.wav"
        subprocess.run(
            [ff, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(concat)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [ff, "-y", "-i", str(concat), "-b:a", bitrate, str(out_mp3)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )


# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--speaker", type=int)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--noise-scale", type=float, default=0.667)
    ap.add_argument("--noise-w", type=float, default=0.8)
    ap.add_argument("--bitrate", default="192k")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--start-page", type=int)
    ap.add_argument("--end-page", type=int)
    args = ap.parse_args()

    inp = Path(args.input)
    if inp.suffix.lower() == ".pdf":
        text = read_text_from_pdf(inp, args.start_page, args.end_page)
    else:
        text = read_text_from_txt(inp)

    text = normalize_headings(normalize_whitespace(text))
    chunks = chunk_text(text, args.max_chars)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        jobs = []
        for i, ch in enumerate(chunks, 1):
            wav = td / f"part_{i:04d}.wav"
            jobs.append((
                i, ch.text, args.model, str(wav),
                args.speaker, args.length_scale,
                args.noise_scale, args.noise_w
            ))

        cpu = os.cpu_count() or 8
        workers = min(args.workers, cpu)
        print(f"[piper] workers={workers} cpu={cpu}")

        results = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_synth_worker, jobs):
                results.append(r)

        results.sort(key=lambda x: x[0])
        wavs = [(Path(w), chunks[i-1].pause_ms) for i, w in results]

        concat_to_mp3(wavs, Path(args.output), args.bitrate)


if __name__ == "__main__":
    main()
