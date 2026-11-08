"""Conversation state for spoken chat with Inky (pure apart from the injected LLM).

Who gets an answer:
  * an utterance that starts with the robot's name (fuzzy, see wake.py), or
  * any utterance inside the follow-up window right after Inky's last chat
    reply finished — so a back-and-forth doesn't need the name every time.
The name alone ("Inky?") gets a short acknowledgement and opens the window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from desktop.voice.wake import WakeMatch, match_wake

DEFAULT_PERSONA = (
    "You are {name}, a small wheeled home robot. You have a depth camera and a lidar; "
    "you map the house, patrol it, and explore it looking for people, pets and toys. "
    "You are talking out loud with the family in their home. Reply in one or two short, "
    "friendly spoken sentences: no lists, no markdown, no emoji, no stage directions. "
    "You can't yet move or act on spoken requests; if asked to, say kindly that you "
    "can't do that by voice yet. If you didn't understand, ask them to say it again."
)

# Whisper's stock outputs for noise/silence; never treat these as a question.
HALLUCINATIONS = {
    "", "you", "thank you", "thanks", "thank you very much", "thanks for watching",
    "thank you for watching", "bye", "okay", "ok", "so", "uh", "um", "hmm", "mm",
}

_PUNCT = re.compile(r"[^a-z0-9' ]+")


def _plain(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


@dataclass
class Decision:
    action: str            # "ignore" | "ack" | "reply"
    reason: str
    prompt: str = ""       # the user's words to answer (reply only)
    wake: Optional[WakeMatch] = None


@dataclass
class Conversation:
    name: str
    llm: Callable[[list[dict]], str]
    persona: str = DEFAULT_PERSONA
    follow_up_s: float = 8.0
    max_turns: int = 6          # user+assistant pairs kept
    reset_after_s: float = 180.0
    ack_text: str = "Yes?"
    history: list[dict] = field(default_factory=list)
    _last_turn: float = 0.0
    _window_until: float = 0.0

    def open_window(self, now: float) -> None:
        """Call when Inky finishes speaking a chat reply."""
        self._window_until = now + self.follow_up_s

    def in_window(self, now: float) -> bool:
        return now < self._window_until

    def decide(self, text: str, now: float) -> Decision:
        text = text.strip()
        wake = match_wake(text, self.name)
        if wake.matched:
            if not _plain(wake.rest) or _plain(wake.rest) in HALLUCINATIONS:
                return Decision("ack", "name only", wake=wake)
            return Decision("reply", "wake word", wake.rest, wake)
        if self.in_window(now):
            if _plain(text) in HALLUCINATIONS:
                return Decision("ignore", "filler in follow-up window", wake=wake)
            return Decision("reply", "follow-up", text, wake)
        return Decision("ignore", "no wake word", wake=wake)

    def messages(self, prompt: str, now: float) -> list[dict]:
        if self.history and now - self._last_turn > self.reset_after_s:
            self.history.clear()
        return ([{"role": "system", "content": self.persona.format(name=self.name)}]
                + self.history + [{"role": "user", "content": prompt}])

    def reply(self, prompt: str, now: float) -> str:
        msgs = self.messages(prompt, now)
        answer = clean_reply(self.llm(msgs))
        self.history += [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}]
        self.history = self.history[-2 * self.max_turns:]
        self._last_turn = now
        self._window_until = 0.0   # reopened when the reply has been spoken
        return answer


def clean_reply(text: str) -> str:
    """Strip what TTS shouldn't read aloud: think blocks, markdown, emoji."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"[*_#`>]+", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)
    return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(text.split()))
