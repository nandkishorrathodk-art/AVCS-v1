"""The closed label vocabulary AVC Lexicon can recognise.

This is deliberately a **closed** set. The reader is a shape classifier, not a
transcriber: it can say "this ink matches the *Save* prototype" and nothing more.
Adding a label means retraining; there is no fallback that spells an unknown
word out, because that fallback would be OCR.

The words are the ones a desktop control is actually labelled with, so the
stream answers the question the grounder needs ("is this the Save button?").
Multi-word labels are kept as single entries with a space; the segmenter is
asked for word boxes, and a two-word label is matched as a two-box sequence.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# 60 labels: the desktop control words this stream is trained to recognise.
LABELS: List[str] = [
    # actions
    "save", "cancel", "ok", "apply", "close", "delete", "edit", "add",
    "remove", "submit", "send", "retry", "undo", "redo", "reset", "refresh",
    # navigation
    "back", "next", "previous", "home", "menu", "settings", "help", "search",
    "file", "edit_menu", "view", "tools", "window",
    # fields and state
    "username", "password", "email", "name", "phone", "address", "city",
    "country", "code", "title", "description", "comment", "message", "subject",
    "yes", "no", "on", "off", "enabled", "disabled", "loading", "error",
    "warning", "success", "failed", "pending", "done", "ready", "upload",
    "download", "open", "install",
]

# "edit" appears twice in spirit (the action and the menu). Keep the vocabulary
# unique: the menu entry is spelled differently above, and this guard makes a
# future duplicate a test failure rather than a silent index collision.
assert len(LABELS) == len(set(LABELS)), "label vocabulary contains duplicates"
assert all(lbl == lbl.strip().lower() for lbl in LABELS), "labels must be lower-case"

VOCAB: Dict[str, int] = {label: i for i, label in enumerate(LABELS)}


def label_to_index(label: str) -> int:
    """Index of a label, or ``-1`` for anything outside the closed vocabulary."""
    return VOCAB.get(label.strip().lower(), -1)


def index_to_label(index: int) -> str:
    """Label for an index; raises on out-of-range so a bad index cannot pass silently."""
    if not 0 <= index < len(LABELS):
        raise IndexError(f"label index {index} outside closed vocabulary of {len(LABELS)}")
    return LABELS[index]


def label_words(label: str) -> Tuple[str, ...]:
    """The word boxes a label is rendered as. A two-word label needs two boxes."""
    return tuple(w for w in label.split() if w)
