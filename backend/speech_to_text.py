"""
Speech-to-Text Module
Converts audio to text using Deepgram REST API
Works with Python 3.14 by using HTTP requests instead of SDK
"""
import os
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeechToText:
    """Converts audio files to text using Deepgram REST API."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Deepgram client using REST API.
        
        Args:
            api_key: Deepgram API key (defaults to config)
        """
        from config import DEEPGRAM_API_KEY
        self.api_key = api_key or DEEPGRAM_API_KEY
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not found. Please set it in api.txt or environment.")
        
        self.base_url = "https://api.deepgram.com/v1/listen"
        logger.info("Deepgram REST client initialized")
    
    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe audio file to text using REST API.
        
        Args:
            audio_path: Path to audio WAV file
            language: Language code (hi=Hindi, en=English)
            
        Returns:
            Transcribed text
        """
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            logger.info(f"Transcribing audio file: {audio_path}")
            
            # Read audio file
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Determine content type - prefer wav for best compatibility
            suffix = audio_path.suffix.lower()
            content_types = {
                '.wav': 'audio/wav',
                '.mp3': 'audio/mpeg',
                '.webm': 'audio/webm',
                '.ogg': 'audio/ogg',
                '.m4a': 'audio/m4a'
            }
            content_type = content_types.get(suffix, 'audio/wav')
            
            # Build URL with query parameters
            # For Hinglish support, use explicit language list for better detection
            if language == "hi,en" or language == "hinglish":
                # Hinglish mode: specify both Hindi and English explicitly
                params = {
                    'model': 'nova-2',
                    'language': 'hi',  # Primary language Hindi
                    'smart_format': 'true',
                    'punctuate': 'true',
                    'encoding': 'linear16' if suffix == '.wav' else None,
                    'sample_rate': 16000
                }
            else:
                params = {
                    'model': 'nova-2',
                    'language': language,
                    'smart_format': 'true',
                    'punctuate': 'true',
                    'encoding': 'linear16' if suffix == '.wav' else None,
                    'sample_rate': 16000
                }
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Build headers
            headers = {
                'Authorization': f'Token {self.api_key}',
                'Content-Type': content_type
            }
            
            # Make API request
            response = requests.post(
                self.base_url,
                params=params,
                headers=headers,
                data=audio_data,
                timeout=30
            )
            
            # Check response status
            if response.status_code != 200:
                error_msg = f"Deepgram API returned {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f": {error_detail}"
                    logger.error(error_msg)
                    
                    # If webm format error, suggest conversion
                    if suffix == '.webm' and 'corrupt' in str(error_detail).lower():
                        logger.warning("WebM format may need conversion. Ensure ffmpeg or pydub is available.")
                except:
                    error_msg += f": {response.text[:200]}"
                    logger.error(error_msg)
                response.raise_for_status()
            
            # Extract transcript
            result = response.json()
            transcript = result.get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('transcript', '')
            
            if not transcript:
                logger.warning("No speech detected in audio")
                return ""
            
            logger.info(f"Transcription successful: {transcript[:50]}...")
            return transcript.strip()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"Deepgram API HTTP error: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response body: {e.response.text[:500]}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Deepgram API request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_multilingual(self, audio_path: str) -> str:
        """
        Transcribe with automatic language detection (Hindi/English).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Try Hindi first
            transcript = self.transcribe(audio_path, language="hi")
            
            # If empty, try English
            if not transcript:
                logger.info("Trying English transcription...")
                transcript = self.transcribe(audio_path, language="en")
            
            return transcript
            
        except Exception as e:
            logger.error(f"Multilingual transcription failed: {e}")
            return ""


if __name__ == "__main__":
    # Test the STT
    logging.basicConfig(level=logging.INFO)
    
    stt = SpeechToText()
    
    # Test with a sample audio file
    test_audio = "test_recording.wav"
    if Path(test_audio).exists():
        result = stt.transcribe_multilingual(test_audio)
        print(f"Transcript: {result}")
    else:
        print(f"Test audio file not found: {test_audio}")
