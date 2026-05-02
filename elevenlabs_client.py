import os
import httpx
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
_TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{_VOICE_ID}"
_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"

_VOICE_SETTINGS = {
    "stability": 0.71,
    "similarity_boost": 0.85,
    "style": 0.12,
    "use_speaker_boost": True,
}


async def text_to_speech(text: str) -> bytes:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            _TTS_URL,
            headers={
                "xi-api-key": _API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": _VOICE_SETTINGS,
            },
        )
        response.raise_for_status()
        return response.content


async def speech_to_text(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            _STT_URL,
            headers={"xi-api-key": _API_KEY},
            data={"model_id": "scribe_v1", "language_code": "en"},
            files={"file": ("audio.webm", audio_bytes, mime_type)},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("text", "").strip()
