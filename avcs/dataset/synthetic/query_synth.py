"""Natural language query synthesis for UI elements and ungroundable negative queries."""
from __future__ import annotations

import random
from typing import List, Optional, Sequence, Set, Tuple

from avcs.query_limits import MAX_QUERY_BYTES


NEGATIVE_QUERY_BANK: List[str] = [
    "Checkout button",
    "Proceed to checkout",
    "Play video",
    "Pause playback",
    "Volume slider",
    "Mute audio",
    "Print document",
    "Download attachment",
    "Upload file",
    "Bluetooth toggle",
    "Wi-Fi networks",
    "Airplane mode",
    "Brightness slider",
    "Microphone mute",
    "Camera shutter",
    "Trash can",
    "Delete account",
    "Log out",
    "Sign in with Google",
    "Shopping cart",
    "Add to cart",
    "Wishlist button",
    "Bookmark this tab",
    "Next song",
    "Previous track",
    "Shuffle playlist",
    "Repeat track",
    "Equalizer settings",
    "Spell check",
    "Insert table",
    "Export as PDF",
    "Run benchmark",
    "Terminal prompt",
    "Clear terminal",
    "Git commit button",
    "Push changes",
    "Merge pull request",
    "Zoom in",
    "Zoom out",
    "Full screen",
    "Dark mode toggle",
    "Help documentation",
    "Send feedback",
    "Share link",
    "Copy URL",
    "Format disk",
    "Restart system",
    "Sleep mode",
    "Lock screen",
    "Battery status",
]


def _truncate_query(query: str, max_bytes: int = MAX_QUERY_BYTES) -> str:
    """Ensures query bytes do not exceed max_bytes."""
    encoded = query.strip().encode("utf-8")
    if len(encoded) <= max_bytes:
        return query.strip()
    # Truncate on character boundary
    decoded = encoded[:max_bytes].decode("utf-8", errors="ignore").strip()
    return decoded or "action"


def synthesize_queries(
    element_kind: str,
    text_label: str,
    action_type: str,
    color_hint: Optional[str] = None,
    position_hint: Optional[str] = None,
    context: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Synthesizes canonical query and rich variations for a UI element.
    
    Returns:
        (canonical_target_query, list_of_query_variations)
    """
    clean_label = text_label.strip()
    variations: List[str] = []

    # 1. Verbatim text label (if meaningful text exists)
    if clean_label and len(clean_label) >= 2 and not clean_label.startswith(("_", "X", "+")):
        variations.append(clean_label)

    # 2. Canonical noun phrase
    kind_phrase = element_kind.replace("_", " ")
    if clean_label:
        canonical = f"{clean_label} {kind_phrase}"
    else:
        canonical = f"{kind_phrase}"
    variations.append(canonical)

    # 3. Action / imperative phrases
    if action_type == "typeable":
        if clean_label:
            variations.append(f"type in {clean_label}")
            variations.append(f"enter text in {clean_label} field")
        variations.append(f"type in {kind_phrase}")
        variations.append(f"click to focus {kind_phrase}")
    else:
        if clean_label:
            variations.append(f"click {clean_label}")
            variations.append(f"press {clean_label}")
            variations.append(f"select {clean_label}")
        else:
            variations.append(f"click {kind_phrase}")
            variations.append(f"press {kind_phrase}")

    # 4. Visual / Color hints
    if color_hint:
        if clean_label:
            variations.append(f"{color_hint} {clean_label} {kind_phrase}")
            variations.append(f"the {color_hint} {clean_label}")
        else:
            variations.append(f"{color_hint} {kind_phrase}")

    # 5. Position / Context hints
    if position_hint:
        if clean_label:
            variations.append(f"{clean_label} {position_hint}")
            variations.append(f"the {clean_label} in the {position_hint}")
        else:
            variations.append(f"{kind_phrase} {position_hint}")

    if context and clean_label:
        variations.append(f"{clean_label} in {context}")

    # Ensure uniqueness, non-empty, and strict byte limits
    seen: Set[str] = set()
    cleaned_variations: List[str] = []
    for q in variations:
        tq = _truncate_query(q)
        if tq and tq.lower() not in seen:
            seen.add(tq.lower())
            cleaned_variations.append(tq)

    target_query = _truncate_query(canonical)
    if not target_query or target_query.lower() not in seen:
        target_query = cleaned_variations[0] if cleaned_variations else "button"

    return target_query, cleaned_variations


def sample_negative_queries(
    present_queries: Sequence[str],
    count: int = 4,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Samples queries guaranteed not to refer to elements on screen."""
    r = rng or random.Random()
    present_words = set()
    for pq in present_queries:
        for w in pq.lower().split():
            if len(w) > 2:
                present_words.add(w)

    candidate_queries = [
        nq for nq in NEGATIVE_QUERY_BANK
        if not any(w in nq.lower() for w in present_words)
    ]
    if len(candidate_queries) < count:
        candidate_queries = [nq for nq in NEGATIVE_QUERY_BANK if nq not in present_queries]

    return r.sample(candidate_queries, min(count, len(candidate_queries)))
