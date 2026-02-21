"""
Handle voice note uploads for speech-to-text and text-to-speech.

Features:
- Voice-to-text transcription via OpenAI Whisper
- Optional voice response via OpenAI TTS
- Duration validation
- Transparent transcription display
"""

import io
from dataclasses import dataclass, field
from typing import Dict, Optional

import structlog
from openai import AsyncOpenAI

from src.config import Settings

logger = structlog.get_logger(__name__)


@dataclass
class ProcessedVoice:
    """Processed voice note result."""

    transcription: str
    prompt: str
    duration_seconds: int
    file_size: int
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class VoiceResponse:
    """Generated voice response."""

    audio_data: bytes
    format: str  # "opus"
    duration_estimate_seconds: float


class VoiceHandler:
    """Process voice note uploads and generate voice responses."""

    def __init__(self, config: Settings):
        self.config = config
        self.max_duration_seconds = config.voice_max_duration_seconds
        self.tts_voice = config.voice_tts_voice
        self.tts_model = config.voice_tts_model
        self.stt_model = config.voice_stt_model
        self.enable_voice_response = config.enable_voice_response
        self.voice_response_max_chars = config.voice_response_max_chars

        # Initialize OpenAI client
        api_key = config.openai_api_key_str
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for voice support")
        self.client = AsyncOpenAI(api_key=api_key)

    async def transcribe_voice(
        self, voice_file_bytes: bytes, duration: int
    ) -> ProcessedVoice:
        """Transcribe a voice note to text.

        Args:
            voice_file_bytes: Raw .ogg bytes from Telegram
            duration: Duration in seconds reported by Telegram

        Returns:
            ProcessedVoice with transcription and prompt

        Raises:
            ValueError: If duration exceeds max or transcription fails
        """
        if duration > self.max_duration_seconds:
            raise ValueError(
                f"Voice note too long ({duration}s). "
                f"Maximum is {self.max_duration_seconds}s."
            )

        logger.info(
            "Transcribing voice note",
            duration=duration,
            file_size=len(voice_file_bytes),
            model=self.stt_model,
        )

        # Whisper accepts .ogg directly — no ffmpeg needed
        audio_file = io.BytesIO(voice_file_bytes)
        audio_file.name = "voice.ogg"

        transcription = await self.client.audio.transcriptions.create(
            model=self.stt_model,
            file=audio_file,
            response_format="text",
        )

        # Handle both string and object responses
        text = (
            transcription.strip()
            if isinstance(transcription, str)
            else transcription.text.strip()
        )

        if not text:
            raise ValueError(
                "Could not transcribe any speech from the voice note."
            )

        logger.info(
            "Voice note transcribed",
            duration=duration,
            transcription_length=len(text),
        )

        prompt = f"[Voice message transcription]: {text}"

        return ProcessedVoice(
            transcription=text,
            prompt=prompt,
            duration_seconds=duration,
            file_size=len(voice_file_bytes),
            metadata={
                "stt_model": self.stt_model,
            },
        )

    async def generate_voice_response(
        self, text: str
    ) -> Optional[VoiceResponse]:
        """Generate a TTS voice response from text.

        Args:
            text: The text to convert to speech

        Returns:
            VoiceResponse with Ogg Opus audio data, or None if disabled/too long
        """
        if not self.enable_voice_response:
            return None

        if len(text) > self.voice_response_max_chars:
            logger.info(
                "Skipping voice response — text too long",
                length=len(text),
                max=self.voice_response_max_chars,
            )
            return None

        logger.info(
            "Generating voice response",
            text_length=len(text),
            voice=self.tts_voice,
            model=self.tts_model,
        )

        response = await self.client.audio.speech.create(
            model=self.tts_model,
            voice=self.tts_voice,
            input=text,
            response_format="opus",  # Ogg Opus — native for Telegram
        )

        audio_data = response.content

        # Rough estimate: ~150 words/min, ~5 chars/word
        estimated_duration = len(text) / (150 * 5 / 60)

        logger.info(
            "Voice response generated",
            audio_size=len(audio_data),
            estimated_duration=round(estimated_duration, 1),
        )

        return VoiceResponse(
            audio_data=audio_data,
            format="opus",
            duration_estimate_seconds=estimated_duration,
        )
