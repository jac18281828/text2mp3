#!/usr/bin/env python3
"""Regression tests for the text2mp3 preprocessing fixes.

Covers the three bug classes from the fix request plus the extras found in the source:
  1a. inline page numbers fused to words (One2 -> One)
  1b. Roman-numeral over-conversion (I'd -> I'500, p.m. -> p.1000)  [now: no body scan]
  2.  two-line CHAPTER headings: keep BOTH marker and spoken "Chapter Five. Murder."
  3.  literal markers: scene breaks, box/rule decoration (= | + ‖), illustrations,
      PG header/footer.

Run:  .venv/bin/python test_text2mp3.py
"""
import os

import text2mp3 as T

_failures = []
_skipped = []


def check(label, got, want):
    if got != want:
        _failures.append(f"{label}\n    got : {got!r}\n    want: {want!r}")


def check_true(label, cond):
    if not cond:
        _failures.append(f"{label}  (expected True)")


# ---------- 1a. page-number stripping (defensive; no-op on the clean .txt) ----------

def test_page_numbers():
    check("One2->One", T.strip_page_numbers("One2 might omit"), "One might omit")
    check("myself4.", T.strip_page_numbers("satisfy myself4."), "satisfy myself.")
    check("He8 ", T.strip_page_numbers("He8 went"), "He went")
    check("presumably)6", T.strip_page_numbers("(presumably)6 so"), "(presumably) so")
    check("that10,", T.strip_page_numbers("saw that10, and"), "saw that, and")
    # legit inline numbers must be preserved
    check("16th-17th", T.strip_page_numbers("the 16th–17th September"), "the 16th–17th September")
    check("time 7.30", T.strip_page_numbers("dine at 7.30?"), "dine at 7.30?")
    check("twenty-one", T.strip_page_numbers("twenty-one"), "twenty-one")
    check("No. 1", T.strip_page_numbers("Souvenirs No. 1, Dr."), "Souvenirs No. 1, Dr.")
    check("(1) paren", T.strip_page_numbers("conclusions. (1) The"), "conclusions. (1) The")
    check("£500", T.strip_page_numbers("£500 of National Savings"), "£500 of National Savings")
    # a fused number above the page range (1-303) is NOT a page number -> keep
    check("over-range", T.strip_page_numbers("room909 now"), "room909 now")


# ---------- 1b. Roman over-conversion is gone (body is left alone) ----------

def test_no_roman_overconversion():
    for probe in ["I’d just like to satisfy myself",
                  "I’m not over sanguine",
                  "You’ll have inquiries made",
                  "at 10 p.m. last night",
                  "Five a.m. I am very tired"]:
        out = T.clean_body_for_tts(probe)
        check(f"roman-safe: {probe!r}", out, probe)
        check_true(f"no 500/1000/100 in {probe!r}",
                   not any(n in out for n in ("500", "1000", "100")))


# ---------- 2. chapter headings: marker + spoken, both present ----------

def test_chapter_headings():
    check("cardinal 5", T.int_to_cardinal_words(5), "five")
    check("cardinal 21", T.int_to_cardinal_words(21), "twenty-one")
    check("cardinal 27", T.int_to_cardinal_words(27), "twenty-seven")
    check("parse V", T.split_chapter_title("CHAPTER V — MURDER"), (5, "MURDER"))
    check("parse XXVII", T.split_chapter_title("CHAPTER XXVII — APOLOGIA"), (27, "APOLOGIA"))
    check("parse no-title", T.split_chapter_title("CHAPTER X")[0], 10)
    check("spoken", T.build_spoken_heading(5, "Murder"), "Chapter Five. Murder.")
    check("spoken 27", T.build_spoken_heading(27, "Apologia"), "Chapter Twenty-seven. Apologia.")
    check("marker", T.build_chapter_marker(5, "Murder"), "Chapter 5: Murder")
    check("smart_title possessive", T.smart_title("WHO’S WHO IN KING’S ABBOT"),
          "Who’s Who in King’s Abbot")
    # A title may open on a quote or bracket; capitalizing the first character
    # rather than the first letter left the word itself lower-cased.
    check("smart_title quoted last word", T.smart_title("THE LETTER SIGNED “BELLA”"),
          "The Letter Signed “Bella”")
    check("smart_title fully quoted", T.smart_title("“SAVE HIM!”"), "“Save Him!”")
    # Leading punctuation is skipped to find the first letter; leading digits are
    # not, or "19th" comes back "19Th".
    check("smart_title ordinal", T.smart_title("THE 19TH CENTURY"), "The 19th Century")
    check("smart_title digit-hyphen", T.smart_title("A 3-DAY JOURNEY"), "A 3-day Journey")
    # No alphabetic character at all: nothing to capitalize, hand it back as-is.
    check("smart_title digits only", T.smart_title("1923"), "1923")


# ---------- 3. literal markers: scene breaks, decoration, illustrations, boilerplate ----------

def test_scene_breaks():
    out = T.clean_body_for_tts("Para one.\n\n* * * * *\n\nPara two.")
    check_true("scene break removed", "*" not in out)
    check_true("paras kept", "Para one." in out and "Para two." in out)


def test_decorations():
    box = (
        "Before.\n\n"
        "  +==============+\n"
        "  ‖  CAPTAIN BLOOD  ‖\n"
        "  ‖==============‖\n"
        "  ‖  THE SEA-HAWK  ‖\n"
        "  +==============+\n\n"
        "After.\n"
    )
    out = T.clean_body_for_tts(box)
    for ch in ("=", "|", "‖", "+"):
        check_true(f"decoration {ch!r} stripped", ch not in out)
    check_true("box text kept", "CAPTAIN BLOOD" in out and "THE SEA-HAWK" in out)
    check_true("surrounding text kept", "Before." in out and "After." in out)


def test_illustrations():
    # multi-line block whose diagram contains stray ']' characters
    block = (
        "Look here.\n\n"
        "[Illustration:\n"
        "  +----+   ]  |\n"
        "  | X  ]  CHAIR |\n"
        "  +----+\n"
        "]\n\n"
        "The butler drew the chair.\n"
    )
    out = T.clean_body_for_tts(block)
    check_true("illustration removed", "Illustration" not in out and "CHAIR" not in out)
    check_true("prose around figure kept", "Look here." in out and "butler drew" in out)
    # single-line illustration
    check_true("single-line illo removed",
               "Illustration" not in T.clean_body_for_tts("a\n\n[Illustration]\n\nb"))


def test_boilerplate():
    doc = (
        "The Project Gutenberg eBook of Something\nTitle: Something\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n\n"
        "Real body text here.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n\n"
        "Section 1. General Terms of Use 1.E.1 license junk\n"
    )
    out = T.strip_gutenberg_boilerplate(doc)
    check("boilerplate", out, "Real body text here.")


def test_front_matter_detection():
    check_true("drop FRONT_MATTER+Contents",
               T._is_front_matter_title("FRONT_MATTER + Contents — Chapter Page"))
    check_true("keep real chapter",
               not T._is_front_matter_title("CHAPTER V — MURDER"))


def test_tiny_toc_keeps_chapter_one():
    # Regression: a tiny TOC must not merge forward into and swallow Chapter 1.
    body1 = "Alpha sentence here. " * 90  # > 1500 chars so it won't merge
    body2 = "Beta sentence here. " * 90
    doc = (
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
        "CONTENTS\n\n  I  FIRST  1\n  II  SECOND  2\n\n"
        "CHAPTER I\n\nFIRST\n\n" + body1 + "\n\n"
        "CHAPTER II\n\nSECOND\n\n" + body2 + "\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n"
    )
    secs = T.prepare_sections(doc, T.BookMetadata(title="X", author="Y"), intro_mode="title")
    titles = [s.title for s in secs]
    check_true("ch1 kept", any(t.startswith("Chapter 1:") for t in titles))
    check_true("ch2 kept", any(t.startswith("Chapter 2:") for t in titles))
    check_true("TOC not narrated", all("SECOND  2" not in s.text for s in secs))


def test_dedication():
    fm = (
        "THE MURDER OF\nROGER ACKROYD\n\nBY\n\nAGATHA CHRISTIE\n\n"
        "Copyright, 1926,\nBy DODD, MEAD AND COMPANY, Inc.\n\n"
        "To Punkie,\nwho likes an orthodox detective\nstory, murder, inquest, and suspicion\n"
        "falling on every one in turn!\n\nCONTENTS\n\n  I  DR. SHEPPARD  1\n"
    )
    ded = T.extract_dedication(fm)
    check("dedication", ded,
          "To Punkie, who likes an orthodox detective story, murder, inquest, "
          "and suspicion falling on every one in turn!")


# ---------- end-to-end on the real file (if present) ----------

def test_real_file_end_to_end():
    path = "TheMurderofRogerckroyd_Agatha_Christie.txt"
    if not os.path.exists(path):
        # The book text is deliberately not in the repo. Say so out loud: a
        # silent return leaves "all 11 test groups passed" claiming coverage
        # this run never had.
        _skipped.append(f"{path} not present")
        return
    raw = open(path, encoding="utf-8", errors="ignore").read()
    meta = T.extract_gutenberg_metadata(raw)
    secs = T.prepare_sections(raw, meta, intro_mode="dedication")
    alltext = "\n".join(s.text for s in secs)
    check("28 sections", len(secs), 28)
    check("first is intro", secs[0].title, "Introduction")
    check("ch5 marker", secs[5].title, "Chapter 5: Murder")
    for ch in ("=", "|", "‖", "+"):
        check_true(f"no {ch!r} in narration", ch not in alltext)
    check_true("no Illustration leak", "Illustration" not in alltext)
    check_true("intro has title+author+dedication",
               "Agatha Christie" in secs[0].text and "Punkie" in secs[0].text)
    check_true("ch5 speaks heading", secs[5].text.startswith("Chapter Five. Murder."))
    check_true("I'd intact", "I’d just like to satisfy myself" in alltext)


# ---------- -m voice-name resolution ----------

class _ParserError(Exception):
    pass


class _RaisingParser:
    """Stands in for argparse: error() raises with the message."""
    def error(self, message):
        raise _ParserError(message)


def _resolve(value, env, cwd=None):
    """Run require_piper_model_files under a patched environment; return the
    resolved path, or the parser error message."""
    keys = ("PIPER_VOICES", "XDG_DATA_HOME", "HOME")
    saved = {k: os.environ.get(k) for k in keys}
    old_cwd = os.getcwd()
    try:
        for k in keys:
            os.environ.pop(k, None)
        os.environ.update(env)
        if cwd:
            os.chdir(cwd)
        try:
            return T.require_piper_model_files(_RaisingParser(), value)
        except _ParserError as e:
            return f"ERROR: {e}"
    finally:
        os.chdir(old_cwd)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _is_error(got):
    return isinstance(got, str) and got.startswith("ERROR:")


def _touch_voice(folder, name, config=True):
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, f"{name}.onnx"), "w").close()
    if config:
        open(os.path.join(folder, f"{name}.onnx.json"), "w").close()


def test_voice_resolution():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        piper = os.path.join(td, "piper")
        xdg = os.path.join(td, "xdg")
        home = os.path.join(td, "home")
        work = os.path.join(td, "work")
        os.makedirs(work)
        home_dir = os.path.join(home, ".local", "share", "piper", "voices")
        xdg_dir = os.path.join(xdg, "piper", "voices")
        for d in (piper, xdg_dir, home_dir):
            _touch_voice(d, "v1")

        # folder precedence: PIPER_VOICES, then XDG_DATA_HOME, then home default
        env = {"PIPER_VOICES": piper, "XDG_DATA_HOME": xdg, "HOME": home}
        check("PIPER_VOICES first", _resolve("v1", env, work),
              Path(piper) / "v1.onnx")
        del env["PIPER_VOICES"]
        check("XDG_DATA_HOME second", _resolve("v1", env, work),
              Path(xdg_dir) / "v1.onnx")
        del env["XDG_DATA_HOME"]
        check("home default last", _resolve("v1", env, work),
              Path(home_dir) / "v1.onnx")

        # a bare name and name.onnx both resolve under the folder
        env = {"PIPER_VOICES": piper}
        check("bare name", _resolve("v1", env, work), Path(piper) / "v1.onnx")
        check("name.onnx", _resolve("v1.onnx", env, work), Path(piper) / "v1.onnx")

        # an existing path wins over a same-named voice
        _touch_voice(work, "v1")
        check("bare file in cwd wins", _resolve("v1.onnx", env, work),
              Path("v1.onnx"))
        local = os.path.join(work, "v1.onnx")
        check("explicit path wins", _resolve(local, env, work), Path(local))

        # a missing voice or a missing config fails, naming the path tried
        got = _resolve("nope", env, work)
        check_true("missing voice fails", _is_error(got))
        check_true("missing voice names path",
                   _is_error(got) and str(Path(piper) / "nope.onnx") in got
                   and piper in got)
        _touch_voice(piper, "noconf", config=False)
        got = _resolve("noconf", env, work)
        check_true("missing config fails", _is_error(got))
        check_true("missing config names path",
                   _is_error(got) and str(Path(piper) / "noconf.onnx.json") in got)
        got = _resolve(os.path.join(work, "gone.onnx"), env, work)
        check_true("missing path keeps message", _is_error(got) and "model file not found" in got)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    total = len(tests)
    if _failures:
        print(f"FAILED ({len(_failures)} checks failed across {total} tests):\n")
        for f in _failures:
            print("  ✗ " + f)
        raise SystemExit(1)
    for reason in _skipped:
        print(f"  skipped: {reason}")
    ran = total - len(_skipped)
    print(f"OK — {ran} of {total} test groups passed"
          + (f", {len(_skipped)} skipped." if _skipped else "."))


if __name__ == "__main__":
    main()
