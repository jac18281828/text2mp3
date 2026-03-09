#!/usr/bin/env python3
"""
text2mp3.py — Offline Piper TTS with smart chapter splitting and multiprocessing
Target: macOS Apple Silicon (M3 Max safe)

Recommended run:
  OMP_NUM_THREADS=1 python3 text2mp3.py ... --workers 6 --split-chapters
"""

from __future__ import annotations

# ===================== Imports =====================

import argparse
import os
import re
import sys
import tempfile
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


# ===================== Text loading =====================


class Text2Mp3Error(Exception):
    """Base class for operational CLI errors."""


class DependencyError(Text2Mp3Error):
    """Raised when an optional runtime dependency is unavailable."""


class SynthesisError(Text2Mp3Error):
    """Raised when Piper synthesis fails."""


_WORKER_VOICE = None
_WORKER_VOICE_MODEL: str | None = None

def read_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_text_from_pdf(path: Path, start_page: int | None, end_page: int | None) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "PDF input requires the `pypdf` package in the active Python environment."
        ) from exc

    reader = PdfReader(str(path))
    n = len(reader.pages)
    s = max(1, start_page) if start_page else 1
    e = min(end_page if end_page else n, n)
    return "\n".join(reader.pages[i].extract_text() or "" for i in range(s - 1, e))


def normalize_whitespace(t: str) -> str:
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def require_existing_file(parser: argparse.ArgumentParser, value: str, label: str) -> Path:
    path = Path(value)
    if not path.exists():
        parser.error(f"{label} not found: {path}")
    if not path.is_file():
        parser.error(f"{label} is not a file: {path}")
    return path


def require_piper_model_files(parser: argparse.ArgumentParser, value: str) -> Path:
    model_path = require_existing_file(parser, value, "model file")
    config_path = Path(f"{model_path}.json")
    if not config_path.exists():
        parser.error(f"model config file not found: {config_path}")
    if not config_path.is_file():
        parser.error(f"model config path is not a file: {config_path}")
    return model_path


# ===================== Metadata extraction =====================

@dataclass
class BookMetadata:
    title: str | None = None
    author: str | None = None
    narrator: str | None = None
    year: str | None = None


def extract_gutenberg_metadata(text: str) -> BookMetadata:
    """
    Extract metadata from Project Gutenberg header.
    Looks for Title:, Author:, Release date: etc. in the first ~100 lines.
    """
    meta = BookMetadata()
    
    # Only scan the header portion (first 100 lines or before START marker)
    lines = text.splitlines()[:100]
    header = "\n".join(lines)
    
    # Check if it's a Gutenberg text
    if "project gutenberg" not in header.lower():
        return meta
    
    # Title: line
    title_match = re.search(r'^Title:\s*(.+)$', header, re.MULTILINE)
    if title_match:
        meta.title = title_match.group(1).strip()
    
    # Author: line
    author_match = re.search(r'^Author:\s*(.+)$', header, re.MULTILINE)
    if author_match:
        meta.author = author_match.group(1).strip()
    
    # Release date: March 1, 2003 [eBook #3810] - extract year
    date_match = re.search(r'^Release date:.*?(\d{4})', header, re.MULTILINE)
    if date_match:
        meta.year = date_match.group(1)
    
    return meta


def extract_metadata_from_filename(path: Path) -> BookMetadata:
    """Fallback: derive title from filename."""
    stem = path.stem
    # Replace underscores with spaces, title case
    title = stem.replace("_", " ").replace("-", " ").strip()
    return BookMetadata(title=title)


def _fix_roman_numerals_case(s: str) -> str:
    """Fix Roman numerals that got lowercased by .title() - e.g. 'Chapter Ii' -> 'Chapter II'"""
    def fix_roman(m):
        word = m.group(0)
        # Check if it's a valid Roman numeral pattern (but was title-cased)
        upper = word.upper()
        if re.fullmatch(r'[IVXLCDM]+', upper):
            return upper
        return word
    return re.sub(r'\b[IVXLCDMivxlcdm]+\b', fix_roman, s)


def normalize_headings(t: str) -> str:
    lines = t.splitlines()
    out = []
    for line in lines:
        s = line.strip()
        if s and len(s) < 80 and s.isupper():
            titled = _fix_roman_numerals_case(s.title())
            # Don't add colon if line already ends with punctuation
            if titled[-1] in '.,:;!?':
                out.append(titled)
            else:
                out.append(titled + ":")
        else:
            out.append(line)
    return "\n".join(out)


# ===================== Roman numeral conversion =====================

_ROMAN_VALUES = [
    ('M', 1000), ('CM', 900), ('D', 500), ('CD', 400),
    ('C', 100), ('XC', 90), ('L', 50), ('XL', 40),
    ('X', 10), ('IX', 9), ('V', 5), ('IV', 4), ('I', 1)
]

def roman_to_int(s: str) -> int | None:
    """Convert a Roman numeral string to integer. Returns None if invalid."""
    s = s.upper()
    result = 0
    idx = 0
    for numeral, value in _ROMAN_VALUES:
        while s[idx:idx+len(numeral)] == numeral:
            result += value
            idx += len(numeral)
    return result if idx == len(s) and result > 0 else None


def _roman_replacer(match: re.Match) -> str:
    """Replace a Roman numeral match with its Arabic equivalent."""
    roman = match.group(0)
    val = roman_to_int(roman)
    if val is not None:
        return str(val)
    return roman


def convert_roman_numerals(t: str) -> str:
    """
    Convert Roman numerals to Arabic numbers for TTS pronunciation.
    Handles standalone Roman numerals (I, II, III, IV, V, ..., MCMLXXXIV, etc.)
    Preserves single 'I' when it's likely the pronoun.
    """
    # Match Roman numerals that are:
    # - Whole words (word boundaries)
    # - Valid Roman numeral characters only
    # - But exclude single 'I' which is usually the pronoun
    
    # Pattern for Roman numerals (case insensitive, but we'll handle case)
    # Must be 2+ chars, OR single char that isn't 'I' (V, X, L, C, D, M)
    roman_pattern = r'\b([MDCLXVI]{2,}|[VXLCDM])\b'
    
    # First pass: uppercase Roman numerals
    t = re.sub(roman_pattern, _roman_replacer, t)
    
    # Second pass: lowercase (less common but possible: "chapter iv")
    t = re.sub(roman_pattern, _roman_replacer, t, flags=re.IGNORECASE)
    
    return t


# ===================== Chunking =====================

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
            buf, blen = [], 0
            for s in smart_sentence_split(p):
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


# ===================== Chapter splitting (robust Gutenberg) =====================

@dataclass
class Section:
    title: str
    text: str

_WORD_NUM = r"(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY)"
_ROMAN_OR_INT = r"([IVXLCDM]+|\d+|" + _WORD_NUM + r")"

# BOOK I CHAPTER I (single line)
_HEADING_INLINE_RE = re.compile(
    rf"^(BOOK|PART)\s+{_ROMAN_OR_INT}\s+CHAPTER\s+{_ROMAN_OR_INT}\b.*$",
    re.IGNORECASE,
)

# Single-line headings (allow optional ":" or "." at end)
_HEADING_LINE_RE = re.compile(
    rf"""^(
        PREFACE|FOREWORD|INTRODUCTION|PROLOGUE|EPILOGUE|
        ACKNOWLEDGMENTS?|DEDICATION|
        CONTENTS|LIST\s+OF\s+ILLUSTRATIONS|
        CHAPTER\s+{_ROMAN_OR_INT}|
        BOOK\s+{_ROMAN_OR_INT}|
        PART\s+{_ROMAN_OR_INT}
    )\s*[:.]?\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

def _is_all_caps_title_line(s: str) -> bool:
    """
    For second-line chapter titles like:
      CHAPTER I
      MY ARRIVAL AT TSAVO
    """
    s = s.strip()
    if not s:
        return False
    if len(s) > 80:
        return False
    # Mostly uppercase letters/spaces/punct
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) >= 0.9


def _looks_like_standalone_heading(lines: list[str], idx: int) -> bool:
    s = lines[idx].strip()
    if not s:
        return False

    prev_blank = idx == 0 or not lines[idx - 1].strip()
    next_blank = idx == len(lines) - 1 or not lines[idx + 1].strip()

    if _HEADING_LINE_RE.match(s):
        return prev_blank or next_blank

    if not _HEADING_INLINE_RE.match(s):
        return False

    if not (prev_blank or next_blank):
        return False

    return len(s) <= 120 and not re.search(r"[,;!?]\s*$", s)

def split_into_sections_smart(text: str, min_section_chars: int = 1500) -> list[Section]:
    """
    Gutenberg-friendly:
    - Detects PREFACE/FOREWORD/etc and CHAPTER/BOOK/PART headings
    - Collapses abutting headings into a single title block
    - Pulls in a 2nd all-caps title line (e.g. 'MY ARRIVAL AT TSAVO')
    """
    lines = text.splitlines()
    n = len(lines)

    blocks: list[tuple[int, int, str]] = []
    i = 0

    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue

        if _looks_like_standalone_heading(lines, i):
            start = i
            titles = [s]

            j = i + 1
            # absorb blank lines + additional heading lines (BOOK I / CHAPTER I abutting)
            while j < n:
                sj = lines[j].strip()
                if not sj:
                    j += 1
                    continue
                if _HEADING_LINE_RE.match(sj):
                    titles.append(sj)
                    j += 1
                    continue
                break

            # If next nonblank line is an all-caps short title, include it as part of heading
            k = j
            while k < n and not lines[k].strip():
                k += 1
            if k < n and _is_all_caps_title_line(lines[k]):
                titles.append(lines[k].strip())
                j = k + 1  # body starts after this title line

            # Dedupe obvious repeats and join
            seen = set()
            norm_titles = []
            for t in titles:
                key = re.sub(r"\s+", " ", t.strip().upper())
                if key not in seen:
                    seen.add(key)
                    # Strip trailing punctuation from title
                    clean = t.strip().rstrip('.:,;')
                    norm_titles.append(clean)
            title = " — ".join(norm_titles)

            blocks.append((start, j, title))
            i = j
        else:
            i += 1

    if not blocks:
        return [Section("FULL_TEXT", text.strip())]

    sections: list[Section] = []
    intro = "\n".join(lines[:blocks[0][0]]).strip()
    if intro:
        sections.append(Section(title="FRONT_MATTER", text=intro))

    for idx, (b_start, b_end, title) in enumerate(blocks):
        next_start = blocks[idx + 1][0] if idx + 1 < len(blocks) else n
        body = "\n".join(lines[b_end:next_start]).strip()
        if body:
            sections.append(Section(title=title, text=body))

    if not sections:
        return [Section("FULL_TEXT", text.strip())]

    # Merge tiny sections forward (stray headings)
    merged: list[Section] = []
    i = 0
    while i < len(sections):
        cur = sections[i]
        if len(cur.text) < min_section_chars and i + 1 < len(sections):
            nxt = sections[i + 1]
            merged.append(Section(
                title=f"{cur.title} + {nxt.title}",
                text=(cur.text + "\n\n" + nxt.text).strip()
            ))
            i += 2
        else:
            merged.append(cur)
            i += 1

    return merged

# ===================== Piper synthesis =====================

def synthesize_chunk_with_piper(text, model, wav, speaker, ls, ns, nw):
    global _WORKER_VOICE, _WORKER_VOICE_MODEL

    model_key = str(Path(model).resolve())
    if _WORKER_VOICE is None or _WORKER_VOICE_MODEL != model_key:
        try:
            from piper import PiperVoice
        except ModuleNotFoundError as exc:
            raise DependencyError(
                f"Piper synthesis dependency is missing from the active Python environment: {exc.name}"
            ) from exc

        try:
            _WORKER_VOICE = PiperVoice.load(model_key)
        except FileNotFoundError as exc:
            missing_path = exc.filename or f"{model_key}.json"
            raise SynthesisError(f"Piper model asset not found: {missing_path}") from exc
        except Exception as exc:
            raise SynthesisError(f"Failed to load Piper model '{model_key}': {exc}") from exc

        _WORKER_VOICE_MODEL = model_key

    try:
        from piper import SynthesisConfig
    except ModuleNotFoundError as exc:
        raise DependencyError(
            f"Piper synthesis dependency is missing from the active Python environment: {exc.name}"
        ) from exc

    syn_config = SynthesisConfig(
        speaker_id=speaker,
        length_scale=ls,
        noise_scale=ns,
        noise_w_scale=nw,
    )

    try:
        with wave.open(str(wav), "wb") as wav_file:
            _WORKER_VOICE.synthesize_wav(text, wav_file, syn_config=syn_config)
    except Exception as exc:
        raise SynthesisError(f"Piper synthesis failed for chunk '{Path(wav).name}': {exc}") from exc


def _synth_worker(job):
    i, text, model, wav, speaker, ls, ns, nw = job
    synthesize_chunk_with_piper(text, model, wav, speaker, ls, ns, nw)
    return i, wav


def run_synthesis_jobs(jobs, workers):
    if workers <= 1:
        return [_synth_worker(job) for job in jobs]

    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_synth_worker, jobs))
    except Text2Mp3Error:
        raise
    except (PermissionError, OSError) as exc:
        if getattr(exc, "errno", None) == 1:
            print("Multiprocessing unavailable; falling back to single-worker synthesis.",
                  file=sys.stderr)
            return [_synth_worker(job) for job in jobs]
        raise SynthesisError(f"Synthesis failed: {exc}") from exc
    except Exception as exc:
        raise SynthesisError(f"Synthesis failed: {exc}") from exc


# ===================== Audio concat =====================

def ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "Audio conversion requires the `imageio-ffmpeg` package in the active Python environment."
        ) from exc

    return imageio_ffmpeg.get_ffmpeg_exe()


def make_silence(out, ms):
    ff = ffmpeg_path()
    subprocess.run(
        [ff, "-y", "-f", "lavfi",
         "-i", "anullsrc=r=22050:cl=mono",
         "-t", f"{ms/1000:.3f}",
         "-c:a", "pcm_s16le", out],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def concat_to_mp3(wavs, out_mp3, bitrate, metadata: BookMetadata | None = None,
                  track_num: int | None = None, track_total: int | None = None,
                  chapter_title: str | None = None):
    ff = ffmpeg_path()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lst, cache = [], {}

        for i, (w, p) in enumerate(wavs):
            lst.append(f"file '{w}'")
            if i < len(wavs) - 1 and p > 0:
                if p not in cache:
                    s = td / f"sil_{p}.wav"
                    make_silence(str(s), p)
                    cache[p] = s
                lst.append(f"file '{cache[p]}'")

        listf = td / "list.txt"
        listf.write_text("\n".join(lst))
        concat = td / "all.wav"

        subprocess.run(
            [ff, "-y", "-f", "concat", "-safe", "0", "-i", listf, "-c", "copy", concat],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Build ffmpeg command with ID3 tags
        cmd = [ff, "-y", "-i", concat, "-b:a", bitrate]
        
        # Add ID3 metadata tags
        if chapter_title:
            cmd.extend(["-metadata", f"title={chapter_title}"])
        elif metadata and metadata.title:
            cmd.extend(["-metadata", f"title={metadata.title}"])
        
        if metadata:
            if metadata.author:
                cmd.extend(["-metadata", f"artist={metadata.author}"])
                cmd.extend(["-metadata", f"album_artist={metadata.author}"])
            if metadata.title:
                cmd.extend(["-metadata", f"album={metadata.title}"])
            if metadata.year:
                cmd.extend(["-metadata", f"date={metadata.year}"])
        
        cmd.extend(["-metadata", "genre=Audiobook"])
        
        if track_num is not None:
            if track_total is not None:
                cmd.extend(["-metadata", f"track={track_num}/{track_total}"])
            else:
                cmd.extend(["-metadata", f"track={track_num}"])
        
        cmd.append(out_mp3)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def write_m3u_playlist(mp3_files: list[tuple[Path, str]], m3u_path: Path):
    """Write an m3u playlist file for the generated mp3s."""
    with open(m3u_path, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for mp3, title in mp3_files:
            # Use relative path from playlist location
            rel = mp3.name
            f.write(f"#EXTINF:-1,{title}\n")
            f.write(f"{rel}\n")
    print(f"Playlist written: {m3u_path}")


def get_wav_duration_ms(wav_path: str) -> int:
    """Get duration of a WAV file in milliseconds using ffprobe."""
    ff = ffmpeg_path()
    # ffprobe is alongside ffmpeg
    ffprobe = str(Path(ff).parent / "ffprobe") if Path(ff).parent.name else "ffprobe"
    # Try using ffmpeg -i to get duration (more reliable than finding ffprobe)
    result = subprocess.run(
        [ff, "-i", wav_path, "-f", "null", "-"],
        capture_output=True, text=True
    )
    # Parse duration from stderr: "Duration: 00:01:23.45"
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if match:
        h, m, s = match.groups()
        return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)
    return 0


def concat_wavs_to_single(wavs: list[tuple[str, int]], out_wav: Path):
    """Concatenate WAV files with silence gaps into a single WAV."""
    ff = ffmpeg_path()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lst, cache = [], {}

        for i, (w, p) in enumerate(wavs):
            lst.append(f"file '{w}'")
            if i < len(wavs) - 1 and p > 0:
                if p not in cache:
                    s = td / f"sil_{p}.wav"
                    make_silence(str(s), p)
                    cache[p] = s
                lst.append(f"file '{cache[p]}'")

        listf = td / "list.txt"
        listf.write_text("\n".join(lst))

        subprocess.run(
            [ff, "-y", "-f", "concat", "-safe", "0", "-i", listf, "-c", "copy", str(out_wav)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )


def create_m4b_audiobook(chapter_wavs: list[tuple[Path, str]], m4b_path: Path, bitrate: str,
                         metadata: BookMetadata | None = None):
    """
    Create an m4b audiobook from chapter WAV files with embedded chapter markers.
    chapter_wavs: list of (wav_path, chapter_title)
    metadata: optional BookMetadata for title/author tags
    """
    try:
        from mutagen.mp4 import MP4
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "Creating audiobook output requires the `mutagen` package in the active Python environment."
        ) from exc

    ff = ffmpeg_path()
    
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        
        # Create concat list and calculate chapter timestamps
        listf = td / "list.txt"
        chapters = []  # (start_ms, end_ms, title)
        current_ms = 0
        
        with open(listf, "w") as f:
            for wav, title in chapter_wavs:
                f.write(f"file '{wav}'\n")
                duration_ms = get_wav_duration_ms(str(wav))
                chapters.append((current_ms, current_ms + duration_ms, title))
                current_ms += duration_ms
        
        # Create ffmpeg metadata file with chapters
        metaf = td / "metadata.txt"
        with open(metaf, "w", encoding="utf-8") as f:
            f.write(";FFMETADATA1\n")
            
            # Book metadata
            book_title = metadata.title if metadata and metadata.title else m4b_path.stem
            f.write(f"title={book_title}\n")
            if metadata and metadata.author:
                f.write(f"artist={metadata.author}\n")
                f.write(f"album_artist={metadata.author}\n")
                f.write(f"composer={metadata.author}\n")
            if metadata and metadata.narrator:
                f.write(f"performer={metadata.narrator}\n")
            if metadata and metadata.year:
                f.write(f"date={metadata.year}\n")
            f.write("genre=Audiobook\n")
            f.write(f"album={book_title}\n")
            f.write("\n")
            
            for start_ms, end_ms, title in chapters:
                # ffmpeg uses milliseconds for chapter timestamps
                f.write("[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={start_ms}\n")
                f.write(f"END={end_ms}\n")
                # Escape special characters in title
                safe_title = title.replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\n", " ")
                f.write(f"title={safe_title}\n\n")
        
        # Concatenate and convert to m4b with chapter metadata
        print(f"Creating audiobook: {m4b_path}")
        if metadata and metadata.title:
            print(f"  Title: {metadata.title}")
        if metadata and metadata.author:
            print(f"  Author: {metadata.author}")
        subprocess.run(
            [ff, "-y", "-f", "concat", "-safe", "0", "-i", str(listf),
             "-i", str(metaf), "-map_metadata", "1",
             "-c:a", "aac", "-b:a", bitrate,
             "-movflags", "+faststart",
             str(m4b_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Set mediakind to Audiobook using mutagen (stik atom)
        # stik value 2 = Audiobook in iTunes/Apple Books
        audio = MP4(str(m4b_path))
        audio["stik"] = [2]  # 2 = Audiobook
        audio.save()
        print(f"  Media kind set to Audiobook")
        
        print(f"Audiobook created: {m4b_path}")


# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--split-chapters", dest="split_chapters", action="store_true", default=True,
                    help="Split into chapters (default)")
    ap.add_argument("--no-split-chapters", dest="split_chapters", action="store_false",
                    help="Disable chapter splitting")
    ap.add_argument("--audiobook", action="store_true",
                    help="Create m4b audiobook file for Apple Books")
    ap.add_argument("--output-pattern")
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--speaker", type=int)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--noise-scale", type=float, default=0.667)
    ap.add_argument("--noise-w", type=float, default=0.8)
    ap.add_argument("--bitrate", default="128k")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start-page", type=int)
    ap.add_argument("--end-page", type=int)
    args = ap.parse_args()

    try:
        inp = require_existing_file(ap, args.input, "input file")
        model_path = require_piper_model_files(ap, args.model)

        if inp.suffix.lower() == ".pdf":
            raw_text = read_text_from_pdf(inp, args.start_page, args.end_page)
        else:
            raw_text = read_text_from_txt(inp)

        # Extract metadata before normalizing text
        metadata = extract_gutenberg_metadata(raw_text)
        if not metadata.title:
            metadata = extract_metadata_from_filename(inp)

        text = normalize_headings(normalize_whitespace(raw_text))
        text = convert_roman_numerals(text)

        # Always detect chapters for metadata, but may not split output files
        sections = split_into_sections_smart(text)

        base = Path(args.output)
        pattern = args.output_pattern or str(base.with_name(base.stem + "_%03d" + base.suffix))

        # For audiobook mode: produce single m4b with chapter markers
        if args.audiobook:
            m4b_path = base.with_suffix(".m4b")

            # Use a persistent temp directory for all chapter WAVs
            with tempfile.TemporaryDirectory() as master_td:
                master_td = Path(master_td)
                chapter_wavs: list[tuple[Path, str]] = []  # (chapter_wav, title)

                for idx, sec in enumerate(sections, 1):
                    print(f"[chapter {idx}/{len(sections)}] {sec.title}")

                    chunks = chunk_text(sec.text, args.max_chars)
                    wavs = []

                    chapter_td = master_td / f"ch_{idx:03d}"
                    chapter_td.mkdir()

                    jobs = []
                    for i, ch in enumerate(chunks, 1):
                        wav = chapter_td / f"p_{i:04d}.wav"
                        jobs.append((i, ch.text, str(model_path), str(wav),
                                     args.speaker, args.length_scale,
                                     args.noise_scale, args.noise_w))

                    workers = min(args.workers, os.cpu_count() or 8)
                    results = run_synthesis_jobs(jobs, workers)

                    results.sort()
                    for i, w in results:
                        wavs.append((w, chunks[i-1].pause_ms))

                    # Concatenate chunks into single chapter WAV
                    chapter_wav = master_td / f"chapter_{idx:03d}.wav"
                    concat_wavs_to_single(wavs, chapter_wav)
                    chapter_wavs.append((chapter_wav, sec.title))

                # Create m4b with chapter metadata
                create_m4b_audiobook(chapter_wavs, m4b_path, args.bitrate, metadata)

        else:
            # Standard mode: produce MP3 files (one per chapter if split, or single file)
            if not args.split_chapters:
                sections = [Section("FULL_TEXT", text)]

            generated_mp3s: list[tuple[Path, str]] = []

            for idx, sec in enumerate(sections, 1):
                if len(sections) == 1 and not args.output_pattern:
                    out_mp3 = base
                else:
                    out_mp3 = Path(pattern % idx)
                print(f"[section {idx}/{len(sections)}] {sec.title}")

                chunks = chunk_text(sec.text, args.max_chars)
                wavs = []

                with tempfile.TemporaryDirectory() as td:
                    td = Path(td)
                    jobs = []
                    for i, ch in enumerate(chunks, 1):
                        wav = td / f"p_{i:04d}.wav"
                        jobs.append((i, ch.text, str(model_path), str(wav),
                                     args.speaker, args.length_scale,
                                     args.noise_scale, args.noise_w))

                    workers = min(args.workers, os.cpu_count() or 8)
                    results = run_synthesis_jobs(jobs, workers)

                    results.sort()
                    for i, w in results:
                        wavs.append((w, chunks[i-1].pause_ms))

                    concat_to_mp3(wavs, out_mp3, args.bitrate,
                                  metadata=metadata,
                                  track_num=idx,
                                  track_total=len(sections),
                                  chapter_title=sec.title)

                generated_mp3s.append((out_mp3, sec.title))

            # Write m3u playlist if we have multiple chapters
            if args.split_chapters and len(generated_mp3s) > 1:
                m3u_path = base.with_suffix(".m3u")
                write_m3u_playlist(generated_mp3s, m3u_path)
    except Text2Mp3Error as exc:
        ap.exit(1, f"Error: {exc}\n")
    except subprocess.CalledProcessError as exc:
        ap.exit(1, f"Error: external command failed with exit code {exc.returncode}: {exc.cmd}\n")
    except FileNotFoundError as exc:
        missing_path = exc.filename or str(exc)
        ap.exit(1, f"Error: file not found: {missing_path}\n")


if __name__ == "__main__":
    main()
