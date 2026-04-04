"""Voice agent — full local pipeline.

Mic → Whisper STT → Kandiga 35B → Qwen3-TTS → Speaker

All local on Apple Silicon. No cloud, no API keys.
Designed for doctor's office phone agent use case.

Usage:
    from kandiga.voice.agent import VoiceAgent
    agent = VoiceAgent()
    agent.start()  # interactive mic/speaker loop
"""

import time
import threading
import numpy as np
import mlx.core as mx


class VoiceAgent:
    """Full local voice agent: STT → LLM → TTS."""

    def __init__(
        self,
        voice: str = "vivian",
        system_prompt: str = None,
    ):
        self.voice = voice
        self.system_prompt = system_prompt or (
            "You are Sarah, the receptionist at Dr. Smith's Family Medicine office. "
            "Office hours are Monday-Friday 8am-5pm. You can schedule appointments, "
            "answer questions, and help callers. You have access to the schedule. "
            "Keep responses SHORT — 1-2 sentences only. Be warm, natural, conversational. "
            "Never say you are an AI. Never refuse to help. Just act like a real receptionist."
        )

        self._stt = None
        self._tts = None
        self._llm = None
        self._ready = False
        self._session_started = False

    def load(self):
        """Load all models. Reports memory at each stage."""
        print("=" * 50)
        print("KANDIGA VOICE AGENT — Loading")
        print("=" * 50)

        t0 = time.time()

        # 1. STT (Whisper)
        from kandiga.voice.stt import VoiceSTT
        self._stt = VoiceSTT()
        self._stt.load()
        mem1 = mx.get_active_memory() / 1e9
        print(f"  GPU after STT: {mem1:.2f} GB")

        # 2. LLM (Kandiga 35B SEM)
        from kandiga.engine import KandigaEngine
        self._llm = KandigaEngine(fast_mode=True)
        self._llm.load()
        mem2 = mx.get_active_memory() / 1e9
        print(f"  GPU after LLM: {mem2:.2f} GB")

        # 3. TTS (Qwen3-TTS)
        from kandiga.voice.tts import VoiceTTS
        self._tts = VoiceTTS()
        self._tts.load(voice=self.voice)
        mem3 = mx.get_active_memory() / 1e9
        print(f"  GPU after TTS: {mem3:.2f} GB")

        total_time = time.time() - t0
        print(f"\n  Total load: {total_time:.1f}s")
        print(f"  Total GPU: {mem3:.2f} GB")
        print(f"  Peak GPU: {mx.get_peak_memory()/1e9:.2f} GB")
        print("=" * 50)

        self._ready = True

    def process_text(self, user_text: str) -> str:
        """Process text input through LLM with persistent KV cache.

        Turn 1: processes system prompt + user message (~2-3s TTFT)
        Turn 2+: only processes new user message (~1-2s TTFT)
        """
        import re

        # First call: start session and inject system prompt
        if not self._session_started:
            self._llm.start_session()
            self._llm._session_history.append({
                "role": "system",
                "content": self.system_prompt,
            })
            self._session_started = True

        # session_generate only processes NEW tokens (KV cache has the rest)
        # Don't break the generator early — it must complete to update
        # session history and token tracking for the next turn
        response_tokens = []
        for token in self._llm.session_generate(
            user_text,
            max_tokens=60,  # short for conversational responses
            temp=0.0,
        ):
            response_tokens.append(token)

        response = "".join(response_tokens).strip()

        # Strip thinking tags
        response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        return response

    def speak(self, text: str):
        """Convert text to speech and play through speakers."""
        import sounddevice as sd

        chunks = []
        for sr, audio in self._tts.speak(text, stream=False):
            chunks.append(audio)

        if chunks:
            full = np.concatenate(chunks)
            audio_f32 = full.astype(np.float32)
            if abs(audio_f32).max() > 1.0:
                audio_f32 = audio_f32 / 32768.0
            sd.play(audio_f32, samplerate=sr)
            sd.wait()

    def listen(self, duration: float = 5.0, sample_rate: int = 16000) -> str:
        """Record from mic and transcribe."""
        import sounddevice as sd

        print("  [Listening...]")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
        )
        sd.wait()
        audio = audio.flatten()

        # Trim silence from the end
        threshold = 0.01
        non_silent = np.where(np.abs(audio) > threshold)[0]
        if len(non_silent) > 0:
            audio = audio[:non_silent[-1] + sample_rate // 2]

        text = self._stt.transcribe(audio, sample_rate)
        return text

    def turn(self, user_text: str = None, listen_duration: float = 5.0):
        """Run one conversation turn: listen → think → speak.

        If user_text is provided, skip listening.
        Returns (user_text, response_text, timing_dict).
        """
        timings = {}

        # STT
        if user_text is None:
            t0 = time.time()
            user_text = self.listen(duration=listen_duration)
            timings['stt'] = time.time() - t0

        print(f"  User: {user_text}")

        # LLM
        t0 = time.time()
        response = self.process_text(user_text)
        timings['llm'] = time.time() - t0
        print(f"  Agent: {response}")

        # TTS
        t0 = time.time()
        self.speak(response)
        timings['tts'] = time.time() - t0

        timings['total'] = sum(timings.values())
        print(f"  [{timings.get('stt', 0):.1f}s STT + {timings['llm']:.1f}s LLM + {timings['tts']:.1f}s TTS = {timings['total']:.1f}s]")

        return user_text, response, timings

    def start(self):
        """Start interactive voice loop. Press Ctrl+C to stop."""
        if not self._ready:
            self.load()

        print("\nVoice agent ready. Speak into mic or type text.")
        print("Type 'q' to quit, or press Enter to listen from mic.\n")

        while True:
            try:
                user_input = input("You (type or press Enter for mic): ").strip()
                if user_input.lower() in ('q', 'quit', 'exit'):
                    break
                if user_input:
                    # Text input — skip STT
                    self.turn(user_text=user_input)
                else:
                    # Mic input
                    self.turn(listen_duration=5.0)
                print()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"  Error: {e}")

        print("\nVoice agent stopped.")


def run_voice_agent():
    """Entry point for CLI."""
    agent = VoiceAgent()
    agent.start()
