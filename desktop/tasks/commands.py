"""Operator command grammar: ``<verb> <object>``, no LLM in the loop.

    goto red ball    go to the dog     come to person
    find red ball    look for the dog  where is the ball
    follow person    follow me
    stop
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

VERBS = ("goto", "find", "follow", "look", "stop")

# Longest phrases first so "go to" wins over "go".
_ALIASES = (
    ("look for", "find"),
    ("search for", "find"),
    ("where is", "find"),
    ("where's", "find"),
    ("go to", "goto"),
    ("come to", "goto"),
    ("drive to", "goto"),
    ("goto", "goto"),
    ("find", "find"),
    ("follow", "follow"),
    ("look at", "look"),
    ("look", "look"),
    ("see", "look"),
    ("stop", "stop"),
    ("halt", "stop"),
    ("cancel", "stop"),
)
_ARTICLES = re.compile(r"^(?:the|a|an|that|this|my)\s+")


@dataclass(frozen=True)
class Command:
    verb: str             # one of VERBS
    target: str = ""      # object name, "" for stop


def parse(text: str) -> Optional[Command]:
    """Parse one command line; None if it isn't one we understand."""
    s = " ".join(text.strip().lower().rstrip("?.!").split())
    if not s:
        return None
    for phrase, verb in _ALIASES:
        if s == phrase or s.startswith(phrase + " "):
            target = _ARTICLES.sub("", s[len(phrase):].strip())
            if verb == "stop":
                return Command("stop")
            if target in ("me", "us"):
                target = "person"
            if not target:
                return None
            return Command(verb, target)
    return None
