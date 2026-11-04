# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Client stop strings, found in the generated text rather than its tokens.

The engine stops on a stop string's own tokenization, but in context the model
often spells the same text with other tokens -- "```" inside a "\\n```" token --
and that match never fires; when it does fire, the stop string's tokens are
already part of the output. The APIs return the text before the first stop
string and nothing after it, so the check is made here on decoded text: what
may still turn out to begin a stop string is held back until it does not, and
the first stop string found ends the text and the request.
"""

from collections.abc import Iterable


def stop_list(stop: str | Iterable[str] | None) -> list[str]:
    """The non-empty stop strings, from either spelling the APIs accept."""
    if stop is None:
        return []
    if isinstance(stop, str):
        stop = [stop]
    return [s for s in stop if s]


def cut_at_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    """``text`` up to its first stop string, and whether there was one."""
    found = [i for i in (text.find(s) for s in stops) if i >= 0]
    return (text[: min(found)], True) if found else (text, False)


class StopStrings:
    """Releases streamed text once it cannot belong to a stop string."""

    def __init__(self, stops: list[str]):
        self.stops = stops
        self.stopped = False
        self._held = ""

    def feed(self, text: str) -> str:
        """What of ``text`` can be released now; nothing after a stop string."""
        if self.stopped:
            return ""
        text, self.stopped = cut_at_stop(self._held + text, self.stops)
        keep = 0 if self.stopped else self._partial_stop(text)
        self._held = text[len(text) - keep :]
        return text[: len(text) - keep]

    def flush(self) -> str:
        """The held-back end, once the text is over without a stop string."""
        text, self._held = self._held, ""
        return text

    def _partial_stop(self, text: str) -> int:
        """Length of the longest end of ``text`` that begins a stop string."""
        return max(
            (
                n
                for stop in self.stops
                for n in range(min(len(stop) - 1, len(text)), 0, -1)
                if text.endswith(stop[:n])
            ),
            default=0,
        )
