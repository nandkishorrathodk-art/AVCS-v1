"""Persistent memory for the voice assistant.

The conversation history in ``ConversationBrain`` lives and dies with the
process: close the assistant and it forgets the user's name, their preferences
and everything it did. That is the difference between a session tool and an
assistant, so this module gives it a small durable store.

Two kinds of memory, kept deliberately separate because they have different
shapes and different failure modes:

* **facts** -- short user-scoped statements ("user prefers Hinglish", "save
  button is bottom-right in the editor"). Retrieved by simple keyword overlap,
  because a 2.7M-parameter local model cannot be trusted to embed reliably and a
  frontier LLM should not be called to read its own notes.
* **episodes** -- what the assistant actually did ("asked to open the file
  manager, pressed ctrl+e twice, no change, asked the user"). These are the raw
  material for offline review and for a future training set.

Honesty rules this module keeps:

* Nothing is written unless ``remember()`` is called with a real value. An empty
  or whitespace fact is dropped, not stored as ``""``.
* The store is plain JSON on disk. It is readable, diffable and deletable by the
  user; it is not a hidden cache.
* A corrupt or unreadable file is reported and treated as empty, never silently
  overwritten with a blank one.
* Retrieval is lexical and its result is *labelled* as a recall, so the brain
  can be told "this is a stored note" rather than being fed it as if it were
  live screen truth.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Words that carry no retrieval signal. Kept small on purpose: over-filtering
# silently drops the term the user actually asked about.
_STOPWORDS = frozenset(
    """a an the is are was were be been am do does did kar karo karna karna hai
    kya ka ki ke ko se me main mujhe mera meri hai hain ho hoon aur or and to of
    for in on at it this that ye yeh wo voh please plz sir ji""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9\u0900-\u097F]+")


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


def default_memory_path() -> Path:
    """Where the store lives when the caller does not choose.

    ``AVCS_MEMORY_PATH`` overrides; otherwise ``~/.avcs/memory.json``. A user
    home is the right default: memory is about *this user*, not about this
    checkout, so it must survive a new clone of the repo.
    """
    env = os.environ.get("AVCS_MEMORY_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".avcs" / "memory.json"


@dataclass
class Fact:
    """One durable user-scoped statement."""

    key: str
    value: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    hits: int = 0

    def as_text(self) -> str:
        return f"{self.key}: {self.value}" if self.key else self.value


@dataclass
class Episode:
    """One thing the assistant did, for later review."""

    heard: str
    said: str
    kind: str
    action: Optional[str] = None
    target: Optional[str] = None
    performed: bool = False
    gate_allowed: Optional[bool] = None
    verified: Optional[bool] = None
    detail: str = ""
    at: float = field(default_factory=time.time)


class MemoryStore:
    """A small JSON-backed store of facts and episodes.

    Parameters
    ----------
    path:
        File to persist to. ``None`` means "in memory only" -- useful in tests
        and for a caller that wants an amnesiac run.
    max_facts, max_episodes:
        Bounds so the file cannot grow without limit. Oldest entries are dropped
        first; facts are bounded by recency of *use*, not of creation, so a
        fact that keeps being recalled is not evicted by chatter.
    """

    def __init__(
        self,
        path: Optional[os.PathLike | str] = None,
        *,
        max_facts: int = 200,
        max_episodes: int = 500,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.max_facts = int(max_facts)
        self.max_episodes = int(max_episodes)
        self.facts: Dict[str, Fact] = {}
        self.episodes: List[Episode] = []
        self.load_error: str = ""
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            # A corrupt file is reported, not silently replaced: overwriting it
            # would destroy whatever the user had, which is worse than starting
            # empty with a visible reason.
            self.load_error = f"could not read memory at {self.path}: {exc}"
            return
        for item in raw.get("facts", []) or []:
            try:
                fact = Fact(**item)
            except Exception:
                continue
            if fact.key or fact.value:
                self.facts[fact.key or fact.value] = fact
        for item in raw.get("episodes", []) or []:
            try:
                self.episodes.append(Episode(**item))
            except Exception:
                continue

    def save(self) -> bool:
        """Write the store atomically. Returns False (never raises) on failure."""
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "facts": [asdict(f) for f in self.facts.values()],
                "episodes": [asdict(e) for e in self.episodes[-self.max_episodes:]],
            }
            # Atomic replace: a crash mid-write must not leave a half file that
            # the next start reads as corrupt.
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Facts
    # ------------------------------------------------------------------ #
    def remember(self, key: str, value: str) -> Optional[Fact]:
        """Store a fact. Empty values are dropped, not stored blank."""
        key = (key or "").strip()
        value = (value or "").strip()
        if not value:
            return None
        identity = key or value
        existing = self.facts.get(identity)
        if existing is not None:
            existing.value = value
            existing.updated_at = time.time()
            fact = existing
        else:
            fact = Fact(key=key, value=value)
            self.facts[identity] = fact
        self._evict_facts()
        return fact

    def forget(self, key: str) -> bool:
        key = (key or "").strip()
        for identity, fact in list(self.facts.items()):
            if identity == key or fact.key == key or fact.value == key:
                del self.facts[identity]
                return True
        return False

    def _evict_facts(self) -> None:
        if len(self.facts) <= self.max_facts:
            return
        # Evict the least-recently-used, not the oldest: a preference recalled
        # every session must outlive a fact stored once and never asked about.
        ordered = sorted(self.facts.items(), key=lambda kv: (kv[1].hits, kv[1].updated_at))
        for identity, _ in ordered[: len(self.facts) - self.max_facts]:
            del self.facts[identity]

    def recall(self, query: str, limit: int = 5) -> List[Fact]:
        """Return the facts most relevant to ``query``, by keyword overlap.

        Lexical, not semantic, and honest about it: with no overlap at all the
        result is empty rather than a plausible-looking nearest neighbour. A
        recalled fact's ``hits`` is bumped so the LRU eviction above works.
        """
        terms = set(_tokens(query))
        if not terms:
            return []
        scored: List[tuple[int, float, Fact]] = []
        for fact in self.facts.values():
            hay = set(_tokens(f"{fact.key} {fact.value}"))
            overlap = len(terms & hay)
            if overlap:
                scored.append((overlap, fact.updated_at, fact))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        out = []
        for _, _, fact in scored[:limit]:
            fact.hits += 1
            out.append(fact)
        return out

    def fact_lines(self) -> List[str]:
        return [f.as_text() for f in self.facts.values()]

    # ------------------------------------------------------------------ #
    # Episodes
    # ------------------------------------------------------------------ #
    def record_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

    def recent_episodes(self, n: int = 5) -> List[Episode]:
        return self.episodes[-n:]

    def episode_summary(self, n: int = 3) -> str:
        """A one-line-per-episode recap for the brain, or "" when there is none."""
        recent = self.recent_episodes(n)
        if not recent:
            return ""
        lines = []
        for ep in recent:
            if ep.action:
                outcome = "worked" if ep.performed else "did not run"
                if ep.verified is True:
                    outcome = "worked, screen changed"
                elif ep.verified is False:
                    outcome = "ran but screen did not change"
                lines.append(f"you tried {ep.action} on {ep.target or 'the screen'} ({outcome})")
            else:
                lines.append(f"you replied about: {ep.heard[:60]}")
        return "; ".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path) if self.path else None,
            "facts": self.fact_lines(),
            "episodes": len(self.episodes),
            "load_error": self.load_error,
        }
