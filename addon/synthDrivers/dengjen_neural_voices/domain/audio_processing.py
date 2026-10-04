# Copyright (c) 2026 Musharraf Omer, Ali Ustek, and contributors
# This file is covered by the GNU General Public License.

"""Audio post-processing utilities for Dengjen Neural Voices.

Provides:
  - AudioStreamProcessor: Stateful processor preserving stream gain & filter continuity
"""

import array
import math

_INT16_MAX = 32767
_INT16_MIN = -32768
_TARGET_RMS = 0.20
_PEAK_CEILING = 0.95

_NIGHT_MODE_GAIN = 0.55
_NIGHT_MODE_HIGHFREQ_DAMP = 0.40


def _pcm_to_array(pcm_bytes: bytes) -> array.array:
    odd = len(pcm_bytes) % 2
    if odd:
        pcm_bytes = pcm_bytes[:-odd]
    a = array.array("h")
    if pcm_bytes:
        a.frombytes(pcm_bytes)
    return a


def _array_to_pcm(a: array.array) -> bytes:
    return a.tobytes()


class AudioStreamProcessor:
    """Stateful audio processor across streaming chunks of an utterance."""

    def __init__(
        self,
        normalize: bool = False,
        night_mode: bool = False,
        spatial_audio: bool = False,
        pan: float = 0.0,
    ):
        self.normalize = normalize
        self.night_mode = night_mode
        self.spatial_audio = spatial_audio
        self.pan = max(-1.0, min(1.0, pan))

        self._remainder = b""
        self._night_mode_prev = 0
        self._norm_gain = None

        angle = (self.pan + 1.0) / 2.0 * (math.pi / 2.0)
        self._left_gain = math.cos(angle)
        self._right_gain = math.sin(angle)

    def process_chunk(self, chunk: bytes) -> bytes:
        if not chunk and not self._remainder:
            return b""

        data = self._remainder + chunk
        if len(data) < 2:
            self._remainder = data
            return b""

        odd = len(data) % 2
        if odd:
            self._remainder = data[-odd:]
            data = data[:-odd]
        else:
            self._remainder = b""

        samples = _pcm_to_array(data)
        if not samples:
            return b""

        # 1. RMS normalization (running gain across chunks to prevent tail amplification)
        if self.normalize:
            n = len(samples)
            sum_sq = sum(s * s for s in samples)
            chunk_rms = math.sqrt(sum_sq / n) / _INT16_MAX

            if chunk_rms >= 0.01:
                target_gain = min(_TARGET_RMS / chunk_rms, 4.0)
                if self._norm_gain is None:
                    self._norm_gain = target_gain
                else:
                    self._norm_gain = 0.85 * self._norm_gain + 0.15 * target_gain
            elif self._norm_gain is None:
                self._norm_gain = 1.0

            gain = self._norm_gain
            ceiling = int(_PEAK_CEILING * _INT16_MAX)
            samples = array.array(
                "h",
                (max(_INT16_MIN, min(ceiling, int(s * gain))) for s in samples),
            )

        # 2. Night mode softening (preserves filter state across chunk boundaries)
        if self.night_mode:
            alpha = _NIGHT_MODE_HIGHFREQ_DAMP
            prev = self._night_mode_prev
            filtered_samples = array.array("h")
            for s in samples:
                filtered = int(alpha * s + (1.0 - alpha) * prev)
                prev = filtered
                out = int(filtered * _NIGHT_MODE_GAIN)
                filtered_samples.append(max(_INT16_MIN, min(_INT16_MAX, out)))
            self._night_mode_prev = prev
            samples = filtered_samples

        # 3. Spatial audio (stereo panning)
        if self.spatial_audio:
            stereo = array.array("h")
            for s in samples:
                left_sample = max(_INT16_MIN, min(_INT16_MAX, int(s * self._left_gain)))
                right_sample = max(
                    _INT16_MIN, min(_INT16_MAX, int(s * self._right_gain))
                )
                stereo.append(left_sample)
                stereo.append(right_sample)
            return _array_to_pcm(stereo)

        return _array_to_pcm(samples)

    def flush(self) -> bytes:
        self._remainder = b""
        return b""
