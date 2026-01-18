"""
Edge TTS Streaming Module
Free and Unlimited TTS using Microsoft Edge neural voices.
"""
import asyncio
import logging
import edge_tts
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

class EdgeTTSStreamer:
    """Handles real-time text-to-speech using Edge TTS (Free & Unlimited)."""
    
    def __init__(self, voice: str = "hi-IN-SwaraNeural"):
        """
        Initialize Edge TTS streamer.
        
        Args:
            voice: Voice ID to use (default is Indian Hindi Female for Hinglish support)
                   Options: hi-IN-SwaraNeural (Female), hi-IN-MadhurNeural (Male)
        """
        self.voice = voice
        logger.info(f"EdgeTTS Streamer initialized (Voice: {self.voice})")

    async def stream_text(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream text to speech and yield audio chunks.
        
        Args:
            text: Text to convert to speech
            
        Yields:
            Audio bytes chunks (MP3 format)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to Edge TTS")
            return
            
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            
            # Buffer chunks to reduce network overhead and choppiness
            chunk_buffer = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunk_buffer.append(chunk["data"])
                    
                    # Yield in larger blocks (e.g., every 16 chunks or ~16KB)
                    if len(chunk_buffer) >= 16:
                        yield b"".join(chunk_buffer)
                        chunk_buffer = []
            
            # Yield remaining
            if chunk_buffer:
                yield b"".join(chunk_buffer)
                            
            logger.info(f"Edge TTS streaming completed for: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Edge TTS streaming failed: {e}")

    async def synthesize_full(self, text: str) -> bytes:
        """Synthesize full audio for text (non-streaming)."""
        chunks = []
        async for chunk in self.stream_text(text):
            chunks.append(chunk)
        return b''.join(chunks)
