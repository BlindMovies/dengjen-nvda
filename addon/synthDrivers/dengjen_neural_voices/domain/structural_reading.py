# Copyright (c) 2026 Musharraf Omer, Ali Ustek, and contributors
# This file is covered by the GNU General Public License.

"""Structural reading: alternate speakers for bracketed/quoted text segments."""

import re
from dataclasses import dataclass

_ASIDE_PATTERNS = [
    re.compile(r"\(([^)]{1,200})\)", re.DOTALL),
    re.compile(r"\[([^\]]{1,200})\]", re.DOTALL),
    re.compile(r'"([^"]{1,200})"', re.DOTALL),
    re.compile(r"「([^」]{1,200})」", re.DOTALL),
]


@dataclass
class SpeechSegment:
    text: str
    speaker_name: str | None = None


def split_into_segments(text: str, default_speaker: str | None = None, alt_speaker: str | None = None) -> list[SpeechSegment]:
    if not text:
        return []

    aside_spans = []
    for pattern in _ASIDE_PATTERNS:
        for m in pattern.finditer(text):
            aside_spans.append((m.start(), m.end()))

    if not aside_spans:
        return [SpeechSegment(text=text, speaker_name=default_speaker)]

    aside_spans.sort()
    merged = [aside_spans[0]]
    for start, end in aside_spans[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    segments: list[SpeechSegment] = []
    cursor = 0
    for aside_start, aside_end in merged:
        if cursor < aside_start:
            main_text = text[cursor:aside_start]
            if main_text.strip():
                segments.append(SpeechSegment(text=main_text, speaker_name=default_speaker))
        aside_text = text[aside_start:aside_end]
        if aside_text.strip():
            segments.append(SpeechSegment(text=aside_text, speaker_name=alt_speaker or default_speaker))
        cursor = aside_end

    if cursor < len(text):
        tail = text[cursor:]
        if tail.strip():
            segments.append(SpeechSegment(text=tail, speaker_name=default_speaker))

    return segments if segments else [SpeechSegment(text=text, speaker_name=default_speaker)]
