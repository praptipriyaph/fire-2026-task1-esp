"""Legal-aware sentence splitter.

The naive regex ``(?<=[.!?])\\s+`` splits on every dot, breaking on common
legal/judicial abbreviations ("Mr. Singh", "accused no. 2", "Section 302.",
"PW-1.", "Hon'ble J.", citations like "AIR 1957 SC 432.") and producing
mid-sentence fragments in the selected ``exact_fact``.

This splitter walks token-by-token. A token ending in ``.!?`` is a candidate
sentence end. We *suppress* the split when:
  - the bare token (lowercase, no trailing punct) is a known abbreviation;
  - the bare token is a single uppercase letter (initial: "S.", "J.");
  - the next token starts with a lowercase letter (continuation);
  - the next token is a legal continuation token ("IPC", "CrPC", "SCC", ...);
  - the bare token is purely digits and the next token starts with lowercase
    (e.g. "...year 1992. the accused..." -> probably not a real boundary).

Run as a script for ad-hoc debugging:
    python sentence_split.py "Mr. Singh, PW-1, deposed that the accused..."
"""

import re
import sys
from typing import List

# Lower-cased bare tokens that should NOT end a sentence.
ABBREV = {
    # courtesy titles
    "mr", "mrs", "ms", "smt", "dr", "shri", "sri", "kum", "prof",
    # judge / advocate titles
    "j", "cj", "aj", "rj", "hon", "hon'ble",
    # legal latin / shorthand
    "v", "vs", "viz", "etc", "cf", "qv", "et", "al", "ie", "eg",
    # ordinals / counters
    "no", "nos", "ors", "anr",
    # citation / section words
    "art", "arts", "sec", "secs", "ch", "ss", "sch",
    "p", "pp", "para", "vol", "ed", "edn", "rev", "supp",
    # Note: citation parts (IPC, CrPC, SCC, AIR, SC, ...) deliberately NOT
    # listed here -- they end real sentences ("...under Section 302 IPC.
    # The court held...") at least as often as they appear inline. The
    # "next-token is digit" and internal-dot rules below catch the inline
    # citation cases.
    # money / units
    "rs", "lacs", "lakh", "crore", "kgs", "gms",
    # time / date
    "am", "pm", "ad", "bc", "ce",
    # latin "with"
    "viz", "qv",
}

# Lower-cased tokens that, if they START the next "sentence", reveal the split
# was a false alarm (continuations of the previous clause).
CONT_START = {
    "ipc", "crpc", "cpc", "scc", "scr", "air", "sc", "j",
    "and", "or", "but", "nor", "yet",
}

# Witness identifiers: PW-1, DW-2, CW-3 etc.
WITNESS_RE = re.compile(r"^(?:p|d|c)w[-_]?\d+\.?$", re.IGNORECASE)
# Combined section citations: 302., 304B., 498A. on their own
SECTION_NUM_RE = re.compile(r"^\d+[A-Za-z]?\.?$")


def _is_sentence_end(tok: str, next_tok: str) -> bool:
    """Return True iff `tok` (ends with .!? somewhere) is a real sentence end."""
    bare = tok.rstrip(".!?").lower()

    if not bare:
        return True

    # Internally-dotted abbreviations: "A.I.R.", "S.C.", "U.P.", "Cr.P.C.".
    # These never end a sentence -- the dots are part of the abbreviation.
    if "." in bare:
        return False

    # Single-letter capital -> initial like "S." in "S. Iyer".
    if len(bare) == 1 and bare.isalpha():
        return False

    if bare in ABBREV:
        return False

    if WITNESS_RE.match(tok):
        return False

    # Next token starts with a digit -> citation continuation
    # ("A.I.R. 1957", "Section 302", "page 45.").
    if next_tok and next_tok[0].isdigit():
        return False

    # Purely-digit token with trailing dot: only a real end if the next
    # word starts with a capital and isn't a continuation cue.
    if bare.isdigit():
        if not next_tok:
            return True
        if next_tok[0].islower():
            return False
        if next_tok.lower().rstrip(",.;:") in CONT_START:
            return False
        return True

    if next_tok:
        nxt = next_tok.lower().rstrip(",.;:)")
        if nxt in CONT_START:
            return False
        # lowercase-led next token is almost always a continuation
        if next_tok[0].islower():
            return False

    return True


def sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    tokens = text.split()
    out, current = [], []
    for i, tok in enumerate(tokens):
        current.append(tok)
        if not re.search(r"[.!?]$", tok):
            continue
        next_tok = tokens[i + 1] if i + 1 < len(tokens) else ""
        if _is_sentence_end(tok, next_tok):
            out.append(" ".join(current))
            current = []
    if current:
        out.append(" ".join(current))
    return out


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: python sentence_split.py \"text to split\"", file=sys.stderr)
        return 2
    text = " ".join(argv[1:])
    for i, s in enumerate(sentences(text), 1):
        print(f"[{i}] {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
