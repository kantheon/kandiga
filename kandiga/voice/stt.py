"""Speech-to-text via Lightning Whisper MLX.

Streaming-capable STT optimized for Apple Silicon.

Usage:
    from kandiga.voice.stt import VoiceSTT
    stt = VoiceSTT()
    stt.load()
    text = stt.transcribe(audio_array)
"""

import time
import numpy as np


# Use distil-large-v3 — best speed/quality on Apple Silicon
DEFAULT_MODEL = "distil-large-v3"


class VoiceSTT:
    """Speech-to-text engine."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self._model_name = model
        self._whisper = None
        self._ready = False

    def load(self):
        """Load Whisper model."""
        t0 = time.time()
        from lightning_whisper_mlx import LightningWhisperMLX
        self._whisper = LightningWhisperMLX(
            model=self._model_name,
            batch_size=12,
            quant=None,
        )
        self._ready = True
        print(f"[voice-stt] Whisper '{self._model_name}' loaded in {time.time()-t0:.1f}s")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text.

        Args:
            audio: float32 numpy array, mono, 16kHz
            sample_rate: sample rate of input audio

        Returns:
            Transcribed text string
        """
        if not self._ready:
            self.load()

        # Resample if needed
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        result = self._whisper.transcribe(audio)
        return result.get("text", "").strip()
