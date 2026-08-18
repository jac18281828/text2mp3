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
import json
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


def clamp_bitrate_for_sample_rate(bitrate: str, sample_rate: int) -> str:
    """MP3 below 32kHz sample rate uses the MPEG-2 LSF format, which tops out
    at 160kbps -- ffmpeg/LAME silently clamp any higher request to it. Match
    what we ask for to what will actually be produced."""
    m = re.match(r"^(\d+)k$", bitrate.strip(), re.IGNORECASE)
    if not m or sample_rate >= 32000:
        return bitrate
    kbps = int(m.group(1))
    if kbps <= 160:
        return bitrate
    print(f"Note: {bitrate} exceeds the 160k MP3 ceiling at {sample_rate}Hz "
          f"(MPEG-2 LSF) -- using 160k, which is what would be produced anyway.")
    return "160k"


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


# ===================== Cardinal words (for spoken chapter headings) =====================

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_WORD_TO_INT = {
    "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7,
    "EIGHT": 8, "NINE": 9, "TEN": 10, "ELEVEN": 11, "TWELVE": 12, "THIRTEEN": 13,
    "FOURTEEN": 14, "FIFTEEN": 15, "SIXTEEN": 16, "SEVENTEEN": 17, "EIGHTEEN": 18,
    "NINETEEN": 19, "TWENTY": 20,
}


def int_to_cardinal_words(n: int) -> str:
    """0-99 -> English words, e.g. 27 -> 'twenty-seven'. Falls back to str() beyond."""
    if n < 0 or n >= 100:
        return str(n)
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"


# ===================== Title casing =====================

_SMALL_WORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "for", "at", "by", "in", "of",
    "on", "to", "up", "as", "is", "it", "with",
}


def _cap_token(w: str) -> str:
    """Capitalize first letter, leave the rest (so possessives stay 'King's', not 'King'S')."""
    return (w[:1].upper() + w[1:]) if w else w


def smart_title(s: str) -> str:
    """Title-case a heading without mangling possessives or shouting small words."""
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return s
    words = s.split(" ")
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if 0 < i < len(words) - 1 and lw in _SMALL_WORDS:
            out.append(lw)
        else:
            out.append(_cap_token(lw))
    return " ".join(out)


# ===================== Gutenberg / TTS text cleaning =====================

def strip_gutenberg_boilerplate(text: str) -> str:
    """Drop everything before '*** START OF ... PROJECT GUTENBERG ...' and from
    '*** END OF ...' onward, removing the PG header and the license footer."""
    start = re.search(r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$",
                      text, re.IGNORECASE | re.MULTILINE)
    end = re.search(r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$",
                    text, re.IGNORECASE | re.MULTILINE)
    s = start.end() if start else 0
    e = end.start() if end else len(text)
    return text[s:e].strip()


def strip_page_numbers(text: str, max_page: int = 303) -> str:
    """Remove inline print page numbers fused to a word with no space (e.g. 'One2',
    'myself4', 'presumably)6'). Only strips a 1-3 digit run that is glued directly to a
    letter or ')', is followed by whitespace/sentence punctuation, and is <= max_page.
    Leaves legitimate inline numbers alone (16th, 7.30, twenty-one, 'No. 1', '(1)')."""
    def repl(m: re.Match) -> str:
        prefix, digits = m.group(1), m.group(2)
        return prefix if int(digits) <= max_page else m.group(0)
    return re.sub(r"([A-Za-z\)])(\d{1,3})(?=[\s.,;:!?]|$)", repl, text)


def strip_scene_breaks(text: str) -> str:
    """Replace asterisk scene-dividers ('***', '* * *', '* * * * *'), markdown rules
    ('---'), and any stray PG START/END lines with a blank line (paragraph break)."""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if s and "*" in s and re.fullmatch(r"[*\s]+", s):
            out.append("")
            continue
        if re.fullmatch(r"-{3,}", s):
            out.append("")
            continue
        if re.match(r"\*\*\*\s*(START|END) OF (?:THE|THIS) PROJECT GUTENBERG", s, re.IGNORECASE):
            out.append("")
            continue
        out.append(line)
    return "\n".join(out)


def strip_italic_underscores(text: str) -> str:
    """Remove the underscore characters Gutenberg uses to mark italics (_word_).

    Where the space after a closing marker was lost in conversion ("_boating_on
    the Brandywine"), deleting the underscore outright would fuse two words into
    one unpronounceable token, so an underscore between two word characters
    becomes a space instead."""
    text = re.sub(r"(?<=\w)_(?=\w)", " ", text)
    return text.replace("_", "")


def strip_illustrations(text: str) -> str:
    """Drop '[Illustration ...]' blocks (figures, ASCII maps/diagrams) — meaningless as
    audio. Handles single-line captions and multi-line blocks that close with a lone ']'
    (the diagrams themselves contain stray ']' characters, so we can't match on ']'
    alone — we terminate only on a line that is just ']')."""
    lines = text.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if s.startswith("[Illustration"):
            # Self-contained on one line, e.g. '[Illustration]' or '[Illustration: x]'
            if s.endswith("]") and s.count("[") <= s.count("]"):
                i += 1
                continue
            # Multi-line block: skip until a line that is exactly ']'
            i += 1
            while i < n and lines[i].strip() != "]":
                i += 1
            i += 1  # consume the closing ']'
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


# Box-drawing / rule characters that must never reach the synthesizer
_DECOR_CHARS = "=+|‖"


def strip_decorations(text: str) -> str:
    """Strip ASCII box-drawing and rule decoration (e.g. the publisher's advertising
    boxes in the back matter) so '=', '|', '‖', '+' are never spoken, while keeping the
    actual text inside the boxes."""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        # Drop lines that are nothing but box-drawing/rule characters
        if s and re.fullmatch(r"[=+\-|‖/\\*.\s]+", s) and re.search(r"[=+|‖]", s):
            out.append("")
            continue
        out.append(line)
    text = "\n".join(out)
    # Remove any residual decoration characters left on content lines (box sides, etc.)
    text = re.sub(r"[=|‖]", " ", text)
    text = re.sub(r"(?<!\w)\++(?!\w)", " ", text)
    return text


def clean_body_for_tts(text: str) -> str:
    """Full body cleanup for narration: illustrations, box decoration, scene breaks,
    page numbers, italics, whitespace."""
    text = strip_illustrations(text)
    text = strip_decorations(text)
    text = strip_scene_breaks(text)
    text = strip_page_numbers(text)
    text = strip_italic_underscores(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ===================== Chapter heading parsing =====================

def parse_chapter_label(token: str) -> int | None:
    """'V' -> 5, '27' -> 27, 'FIVE' -> 5. Returns None if not a chapter number."""
    token = token.strip().rstrip(".:").upper()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    roman = roman_to_int(token)
    if roman is not None:
        return roman
    return _WORD_TO_INT.get(token)


def split_chapter_title(full_title: str) -> tuple[int | None, str]:
    """Parse a section title like 'CHAPTER V — MURDER' into (5, 'MURDER').
    Returns (None, full_title) when there is no CHAPTER/BOOK/PART label."""
    parts = re.split(r"\s*—\s*", full_title)
    head = parts[0].strip()
    m = re.match(r"^(?:CHAPTER|BOOK|PART)\s+(.+)$", head, re.IGNORECASE)
    if not m:
        return None, full_title
    num = parse_chapter_label(m.group(1))
    title = " ".join(p.strip() for p in parts[1:]).strip()
    return num, title


def build_spoken_heading(num: int, title: str) -> str:
    """Spoken chapter intro, e.g. (5, 'Murder') -> 'Chapter Five. Murder.'"""
    words = int_to_cardinal_words(num).capitalize()
    if title:
        return f"Chapter {words}. {ensure_terminal_punct(title)}"
    return f"Chapter {words}."


def build_chapter_marker(num: int, title: str) -> str:
    """m4b/ID3 chapter marker, e.g. (5, 'Murder') -> 'Chapter 5: Murder'."""
    return f"Chapter {num}: {title}" if title else f"Chapter {num}"


def extract_dedication(text: str) -> str | None:
    """Pull the dedication (last paragraph before CONTENTS) out of front matter."""
    m = re.search(r"(.*?)\n\s*CONTENTS\b", text, re.IGNORECASE | re.DOTALL)
    region = m.group(1) if m else text[:4000]
    paras = [p.strip() for p in re.split(r"\n\s*\n", region) if p.strip()]
    if not paras:
        return None
    ded = paras[-1]
    if re.search(r"GUTENBERG|COPYRIGHT|DODD|GROSSET|PUBLISHERS|ILLUSTRATION|AUTHOR OF",
                 ded, re.IGNORECASE):
        return None
    return re.sub(r"\s+", " ", ded).strip()


def build_intro(metadata: "BookMetadata", dedication: str | None, mode: str) -> str:
    """Compose the spoken opening that replaces the PG boilerplate / title page / TOC."""
    title = smart_title(metadata.title) if metadata.title else "This book"
    if metadata.author:
        parts = [f"{title}. By {metadata.author}."]
    else:
        parts = [f"{title}."]
    if mode == "dedication" and dedication:
        parts.append(dedication)
    elif mode == "note":
        parts.append("This is a synthetic text-to-speech reading, produced from the "
                     "public-domain Project Gutenberg edition.")
    return PARA_JOIN.join(parts)


def _is_front_matter_title(title: str) -> bool:
    """True for front-matter/TOC sections that should be dropped from narration.
    Real chapter titles never contain CONTENTS/FRONT_MATTER, but a merged front-matter
    block can contain the word CHAPTER (from the 'CHAPTER ... PAGE' TOC header), so we
    key off the front-matter markers directly rather than the presence of CHAPTER."""
    t = title.upper()
    return any(k in t for k in ("FRONT_MATTER", "CONTENTS", "ILLUSTRATIONS", "LIST OF"))


# Headings that mark the start of actual narrative content (NOT front matter such as
# CONTENTS / DEDICATION / LIST OF ILLUSTRATIONS).
_CONTENT_HEADING_RE = re.compile(
    r"^(CHAPTER|BOOK|PART|PROLOGUE|PREFACE|FOREWORD|INTRODUCTION)\b", re.IGNORECASE)


def _find_narrative_start(lines: list[str]) -> int:
    """Index of the first standalone *content* heading. Everything before it (title page,
    table of contents, dedication) is front matter. Returns 0 if none is found."""
    for i in range(len(lines)):
        if _looks_like_standalone_heading(lines, i) and \
                _CONTENT_HEADING_RE.match(lines[i].strip()):
            return i
    return 0


def prepare_sections(raw_text: str, metadata: "BookMetadata",
                     intro_mode: str = "title") -> list["Section"]:
    """Build narration-ready sections: strip boilerplate, drop front matter/TOC,
    give each chapter a spoken heading + clean marker, and clean each body."""
    stripped = strip_gutenberg_boilerplate(raw_text)
    norm = re.sub(r"[ \t]+", " ", stripped)
    norm = re.sub(r"\n{3,}", "\n\n", norm).strip()

    # Separate front matter (title page / TOC / dedication) from the narrative BEFORE
    # splitting, so a tiny TOC can never merge forward into (and swallow) chapter 1.
    lines = norm.split("\n")
    start = _find_narrative_start(lines)
    front_text = "\n".join(lines[:start])
    narrative = "\n".join(lines[start:]).strip() or norm

    raw_sections = split_into_sections_smart(narrative)

    out: list[Section] = []
    if metadata.title:
        dedication = extract_dedication(front_text or norm) if intro_mode != "none" else None
        out.append(Section(title="Introduction",
                            text=build_intro(metadata, dedication, intro_mode)))

    for sec in raw_sections:
        if _is_front_matter_title(sec.title):
            continue
        num, ctitle = split_chapter_title(sec.title)
        body = clean_body_for_tts(sec.text)
        if num is not None:
            disp = smart_title(ctitle) if ctitle else ""
            spoken = build_spoken_heading(num, disp)
            marker = build_chapter_marker(num, disp)
            text = spoken + "\n\n" + body if body else spoken
            out.append(Section(title=marker, text=text))
        else:
            out.append(Section(title=smart_title(sec.title), text=body))

    return out if out else [Section("FULL_TEXT", clean_body_for_tts(norm))]


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


# Paragraphs packed into one chunk are joined with a single newline, never a
# blank line. A blank line inside the text handed to Piper makes it swallow the
# word immediately before it -- deterministically, on every repeat, and whatever
# the punctuation (comma, period, or none). Measured on the Cori voice: three
# verse lines joined by "\n\n" lost two of their three line-final words 3/3
# times; the same lines joined by "\n" kept all of them 3/3 times. A single
# newline still reads as a phrase break, so nothing is lost by using it.
PARA_JOIN = "\n"


def join_paragraphs(t: str) -> str:
    """Collapse paragraph breaks to the separator Piper handles safely."""
    return PARA_JOIN.join(p.strip() for p in t.split("\n\n") if p.strip())


def chunk_text(t: str, max_chars: int) -> list[Chunk]:
    if len(t) <= max_chars:
        return [Chunk(ensure_terminal_punct(join_paragraphs(t)), 700)]

    paras = [p.strip() for p in t.split("\n\n") if p.strip()]
    chunks, cur, cur_len = [], [], 0

    def flush(pause):
        nonlocal cur, cur_len
        if cur:
            chunks.append(Chunk(ensure_terminal_punct(PARA_JOIN.join(cur)), pause))
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
                    if buf:
                        chunks.append(Chunk(ensure_terminal_punct(" ".join(buf)), 220))
                    buf, blen = [s], len(s)
            if buf:
                chunks.append(Chunk(ensure_terminal_punct(" ".join(buf)), 700))
        else:
            cur, cur_len = [p], len(p)

    flush(700)
    # Drop chunks with nothing worth speaking: sentence-splitting can strand a
    # bare punctuation fragment (e.g. a lone closing quote from "...all.'")
    # as its own "sentence"; ensure_terminal_punct then pads it to something
    # like "'." -- non-blank, but no actual words, and Piper crashes on it
    # (writes zero audio frames, never initializes the WAV header).
    return [c for c in chunks if re.search(r"\w", c.text)]


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

def _apply_edge_fades(wav_path, fade_ms=10):
    """Fade the first/last `fade_ms` of a synthesized chunk to silence. Piper's
    raw output rarely ends on a zero-crossing, so splicing chunks back to back
    with a straight `ffmpeg -c copy` concat produces an audible click at every
    boundary; a short fade at each edge makes the cut inaudible. Fades both
    ends without needing the clip's duration up front (fade-in, reverse,
    fade-in again, reverse back)."""
    ff = ffmpeg_path()
    fade_s = fade_ms / 1000
    tmp = f"{wav_path}.faded.wav"
    subprocess.run(
        [ff, "-y", "-i", str(wav_path),
         "-af", f"afade=t=in:st=0:d={fade_s},areverse,"
                f"afade=t=in:st=0:d={fade_s},areverse",
         "-c:a", "pcm_s16le", tmp],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    os.replace(tmp, wav_path)


def synthesize_chunk_with_piper(text, model, wav, speaker, ls, ns, nw, volume=1.0):
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
        normalize_audio=False,
        volume=volume,
    )

    try:
        with wave.open(str(wav), "wb") as wav_file:
            _WORKER_VOICE.synthesize_wav(text, wav_file, syn_config=syn_config)
        _apply_edge_fades(wav)
    except Exception as exc:
        raise SynthesisError(f"Piper synthesis failed for chunk '{Path(wav).name}': {exc}") from exc


def _synth_worker(job):
    i, text, model, wav, speaker, ls, ns, nw, volume = job
    synthesize_chunk_with_piper(text, model, wav, speaker, ls, ns, nw, volume)
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


def sanitize_filename(name: str) -> str:
    """Make a chapter title safe for a filename (keep it readable)."""
    name = name.replace(":", " -")
    name = re.sub(r'[\\/*?"<>|]', "", name)
    return re.sub(r"\s+", " ", name).strip()


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
    ap.add_argument("--intro", choices=["title", "dedication", "note", "none"],
                    default="title",
                    help="Spoken opening that replaces the Gutenberg boilerplate/title "
                         "page/TOC (default: title + author)")
    ap.add_argument("--also-mp3", action="store_true",
                    help="In --audiobook mode, also write per-chapter MP3s + .m3u "
                         "(for the archive.org streaming player) from the same audio")
    ap.add_argument("--mp3-dir",
                    help="Directory for per-chapter MP3s (default: <output stem>_mp3)")
    ap.add_argument("--title",
                    help="Book title for the spoken intro and file tags. Overrides "
                         "whatever is inferred from the Gutenberg header or filename, "
                         "which loses leading articles and subtitles.")
    ap.add_argument("--author",
                    help="Author for the spoken intro and file tags.")
    ap.add_argument("--output-pattern")
    ap.add_argument("--max-chars", type=int, default=600,
                    help="Chunk size in characters (default: 600). Larger chunks "
                         "mean fewer splices; the dropped-word problem this used "
                         "to trade against was a blank line inside chunk text (see "
                         "PARA_JOIN), not chunk length, and is fixed at the source.")
    ap.add_argument("--speaker", type=int)
    ap.add_argument("--length-scale", type=float, default=1.20)
    ap.add_argument("--noise-scale", type=float, default=0.6)
    ap.add_argument("--noise-w", type=float, default=0.75)
    ap.add_argument("--volume", type=float, default=1.4,
                    help="Output gain multiplier applied during synthesis "
                         "(default: 1.4). Only relevant since normalize_audio "
                         "is off -- Piper no longer peak-normalizes each chunk, "
                         "so overall level is quieter than before and this is "
                         "the way to bring it back up.")
    ap.add_argument("--bitrate", default="160k",
                    help="MP3 bitrate (default: 160k). Voices under 32kHz sample "
                         "rate (most Piper voices, incl. all bundled here) top out "
                         "at 160k -- MP3's MPEG-2 LSF format has no higher option "
                         "at that sample rate, so anything above 160k gets clamped.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start-page", type=int)
    ap.add_argument("--end-page", type=int)
    args = ap.parse_args()

    try:
        inp = require_existing_file(ap, args.input, "input file")
        model_path = require_piper_model_files(ap, args.model)

        with open(f"{model_path}.json", encoding="utf-8") as f:
            voice_sample_rate = json.load(f).get("audio", {}).get("sample_rate", 22050)
        args.bitrate = clamp_bitrate_for_sample_rate(args.bitrate, voice_sample_rate)

        if inp.suffix.lower() == ".pdf":
            raw_text = read_text_from_pdf(inp, args.start_page, args.end_page)
        else:
            raw_text = read_text_from_txt(inp)

        # Extract metadata before stripping the Gutenberg header
        metadata = extract_gutenberg_metadata(raw_text)
        if not metadata.title:
            metadata = extract_metadata_from_filename(inp)
        if args.title:
            metadata.title = args.title
        if args.author:
            metadata.author = args.author

        # Build narration-ready sections: strip PG boilerplate/TOC, give each chapter a
        # spoken heading + clean marker, clean each body. Roman->number is applied ONLY
        # to chapter labels here (no destructive global body scan).
        sections = prepare_sections(raw_text, metadata, intro_mode=args.intro)

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
                                     args.noise_scale, args.noise_w, args.volume))

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

                # Optionally emit per-chapter MP3s + m3u from the SAME synthesized WAVs
                # (no re-synthesis), for the archive.org inline streaming player.
                if args.also_mp3:
                    mp3_dir = Path(args.mp3_dir) if args.mp3_dir else \
                        base.with_name(base.stem + "_mp3")
                    mp3_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Writing per-chapter MP3s to: {mp3_dir}")
                    total = len(chapter_wavs)
                    generated_mp3s: list[tuple[Path, str]] = []
                    for idx, (chapter_wav, title) in enumerate(chapter_wavs, 1):
                        out_mp3 = mp3_dir / f"{idx:02d} - {sanitize_filename(title)}.mp3"
                        concat_to_mp3([(str(chapter_wav), 0)], str(out_mp3), args.bitrate,
                                      metadata=metadata, track_num=idx, track_total=total,
                                      chapter_title=title)
                        generated_mp3s.append((out_mp3, title))
                        print(f"  [mp3 {idx}/{total}] {out_mp3.name}")
                    write_m3u_playlist(generated_mp3s, mp3_dir / f"{base.stem}.m3u")

        else:
            # Standard mode: produce MP3 files (one per chapter if split, or single file)
            if not args.split_chapters:
                full_text = "\n\n".join(s.text for s in sections)
                sections = [Section("FULL_TEXT", full_text)]

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
                                     args.noise_scale, args.noise_w, args.volume))

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
