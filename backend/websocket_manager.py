import json
import logging
import asyncio
from fastapi import WebSocket
from typing import List, Dict, Optional
import base64
import time
import requests

from stt.deepgram_stream import DeepgramStreamer
from tts.elevenlabs_stream import ElevenLabsStreamer
from llm.gemini_client import GeminiClient
from gemini_llm import GeminiLLM
from groq_llm import GroqLLM
from tts.edge_tts_streamer import EdgeTTSStreamer
from memory import ConversationMemory
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')


def get_fresh_keys():
    """Refresh environment variables from .env and api.txt file."""
    from dotenv import load_dotenv
    from pathlib import Path
    import os
    
    # Reload .env
    load_dotenv(override=True)
    
    # Check api.txt in parent directory
    api_file = Path(__file__).parent.parent / "api.txt"
    api_keys = {}
    if api_file.exists():
        try:
            with open(api_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        k, v = line.split('=', 1)
                        key = k.strip()
                        value = v.strip()
                        if key == "Deepgram":
                            api_keys['DEEPGRAM_API_KEY'] = value
                        elif key == "Eleven Labs":
                            api_keys['ELEVENLABS_API_KEY'] = value
        except:
            pass
            
    dg_key = os.getenv('DEEPGRAM_API_KEY') or api_keys.get('DEEPGRAM_API_KEY')
    el_key = os.getenv('ELEVENLABS_API_KEY') or api_keys.get('ELEVENLABS_API_KEY')
    groq_key = os.getenv('GROQ_API_KEY') or api_keys.get('GROQ_API_KEY')
    voice_id = os.getenv('ELEVENLABS_VOICE_ID')
    return dg_key, el_key, voice_id, groq_key


def calculate_amplitude(base64_audio: str) -> float:
    """Calculate audio amplitude from base64 encoded audio for visual feedback."""
    try:
        import base64
        audio_bytes = base64.b64decode(base64_audio)
        # Calculate RMS amplitude from audio bytes
        if len(audio_bytes) < 2:
            return 0.0
        samples = [int.from_bytes(audio_bytes[i:i+2], 'little', signed=True) 
                   for i in range(0, min(len(audio_bytes), 1000), 2)]
        if not samples:
            return 0.0
        rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
        # Normalize to 0-1 range
        return min(1.0, rms / 32768.0 * 10)
    except Exception:
        return 0.5  # Default mid-level if calculation fails

class WebSocketManager:
    """Manages WebSocket connections and coordinates STT, LLM, and TTS."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        # Audio buffering for REST API fallback (since streaming STT is disabled)
        self.audio_buffer: Dict[str, List[bytes]] = {}  # websocket id -> audio chunks
        self.session_headers: Dict[str, bytes] = {}     # websocket id -> first chunk (with header)
        self.buffer_start_time: Dict[str, float] = {}
        self.is_recording: Dict[str, bool] = {}
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket Client connected: {websocket.client}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Clean up audio buffers for this connection
        ws_id = str(id(websocket))
        if ws_id in self.audio_buffer:
            del self.audio_buffer[ws_id]
        if ws_id in self.buffer_start_time:
            del self.buffer_start_time[ws_id]
        if ws_id in self.is_recording:
            del self.is_recording[ws_id]
        if ws_id in self.session_headers:
            del self.session_headers[ws_id]
            
        logger.info(f"WebSocket Client disconnected: {websocket.client}")
    
    async def broadcast_metrics(self):
        """Periodically broadcast system metrics to all connected clients."""
        from system_monitor import system_monitor
        while True:
            if self.active_connections:
                try:
                    metrics = system_monitor.get_system_info()
                    for websocket in list(self.active_connections):
                        try:
                            await websocket.send_json({
                                "type": "system_metrics",
                                "data": metrics
                            })
                        except Exception as e:
                            logger.debug(f"Error sending metrics to a client: {e}")
                except Exception as e:
                    logger.error(f"Error in broadcast_metrics: {e}")
            await asyncio.sleep(3)  # Update every 3 seconds

    async def handle_voice_session(self, websocket: WebSocket):
        """Main loop for coordinating real-time voice interaction and text chat."""
        
        # Initialize Streamers (only for voice)
        stt_streamer = None
        tts_streamer = None
        
        # Get fresh keys
        _, fresh_eleven_key, fresh_voice_id, fresh_groq_key = get_fresh_keys()
        
        # Determine which TTS to use (Strict Priority: ElevenLabs -> EdgeTTS)
        tts_engine = "ElevenLabs"
        try:
            # Check for ElevenLabs key and ensure it's not a placeholder
            if fresh_eleven_key and len(fresh_eleven_key) > 10 and not fresh_eleven_key.startswith("sk_placeholder"):
                logger.info(f"Attempting to initialize ElevenLabs with Voice ID: {fresh_voice_id}")
                tts_streamer = ElevenLabsStreamer(fresh_eleven_key, voice_id=fresh_voice_id)
                logger.info(f"[OK] SUCCESSFULLY initialized ElevenLabs premium voice ({fresh_voice_id})")
                await websocket.send_json({"type": "status_info", "engine": "ElevenLabs Premium"})
            else:
                logger.warning("ElevenLabs key missing or invalid. Falling back to EdgeTTS.")
                tts_streamer = EdgeTTSStreamer()
                tts_engine = "EdgeTTS"
                logger.info("Using EdgeTTS (Free Fallback)")
                await websocket.send_json({"type": "status_info", "engine": "EdgeTTS (Free)"})
        except Exception as e:
            logger.error(f"[ERROR] ElevenLabs init failed: {e}. Falling back to EdgeTTS.")
            tts_streamer = EdgeTTSStreamer()
            tts_engine = "EdgeTTS"
            await websocket.send_json({"type": "status_info", "engine": "EdgeTTS (Emergency Fallback)"})

        # Create isolated memory for this session
        session_memory = ConversationMemory()
        
        # Select and initialize the active LLM for this session
        if fresh_groq_key and len(fresh_groq_key) > 20: 
            try:
                active_llm = GroqLLM(api_key=fresh_groq_key, memory=session_memory)
                logger.info("Using Groq as active LLM for this session")
            except Exception as e:
                logger.error(f"Groq init failed, falling back: {e}")
                active_llm = GeminiLLM(memory=session_memory)
        else:
            active_llm = GeminiLLM(memory=session_memory)
            logger.info("Using Gemini as active LLM for this session")
        
        # Create a new conversation memory for this WebSocket session
        # Note: Memory is now handled internally by self.gemini_llm

        async def on_transcript_callback(transcript: str, is_final: bool):
            """Callback for STT results."""
            if is_final:
                await websocket.send_json({"type": "final_transcript", "payload": transcript})
                
                logger.info("\n" + "="*50)
                logger.info("[MIC] Listening...")
                logger.info("Done listening")
                
                # Transcribing
                transcribe_start = time.time()
                logger.info("[TRANSCRIBE] Transcribing audio...")
                # (Transcription happens in callback)
                transcribe_time = time.time() - transcribe_start
                logger.info(f"[OK] Finished transcribing in {transcribe_time:.2f} seconds.")
                
                # Initialize Task Planner (The "Brain")
                try:
                    from automation.task_planner import TaskPlanner
                    # Pass the active LLM to the planner
                    planner = TaskPlanner(llm_client=active_llm)
                except Exception as e:
                    logger.error(f"Failed to init TaskPlanner: {e}")
                    planner = None

                # Generate response
                response_start = time.time()
                logger.info(f"[AI] Planning & Executing for: '{transcript[:50]}...'")
                
                # 1. ACTION FIRST: Try to Execute Plan
                is_executed = False
                ai_response = ""
                
                if planner:
                    try:
                        plan_result = await planner.execute_plan(transcript)
                        if plan_result != "NO_ACTION_REQUIRED":
                            ai_response = plan_result
                            is_executed = True
                            logger.info(f"[EXEC] Executed Plan: {ai_response}")
                    except Exception as e:
                         logger.error(f"Plan Execution Failed: {e}")
                
                # 2. IF NO ACTION: Do normal chat (Knowledge Mode)
                if not is_executed or not ai_response:
                    ai_response = await asyncio.to_thread(active_llm.generate_response, transcript)
                                
                response_time = time.time() - response_start
                logger.info(f"[OK] Finished generating response in {response_time:.2f} seconds.")
                
                # Send AI response (using frontend-expected format)
                await websocket.send_json({
                    "type": "response",
                    "content": ai_response
                })
                
                # Send to TTS if available
                if tts_streamer:
                    try:
                        audio_start = time.time()
                        logger.info("[AUDIO] Generating audio...")
                        async for audio_chunk in tts_streamer.stream_text(ai_response):
                            audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')
                            # Essential: Send level for orb animation
                            level = calculate_amplitude(audio_base64)
                            await websocket.send_json({"type": "audio_level", "payload": level})
                            
                            await websocket.send_json({
                                "type": "tts_audio",
                                "payload": audio_base64
                            })
                        audio_time = time.time() - audio_start
                        logger.info(f"[OK] Finished generating audio in {audio_time:.2f} seconds.")
                        
                        logger.info("[SPEAK] Speaking...")
                    except Exception as e:
                        logger.error(f"TTS streaming failed: {e}")
                
                # Send status update
                await websocket.send_json({"type": "status", "status": "idle"})
                logger.info("="*50 + "\n")

        def tts_audio_callback(base64_audio: str):
            """Callback for TTS audio chunks."""
            # Calculate amplitude for reactive orb
            level = calculate_amplitude(base64_audio)
            asyncio.create_task(websocket.send_json({"type": "audio_level", "payload": level}))
            asyncio.create_task(websocket.send_json({"type": "tts_audio", "payload": base64_audio}))

        # Internal helper for tts callback because closures/async 
        self._tts_callback = tts_audio_callback

        # Initialize STT only if Deepgram API key is available
        # NOTE: Deepgram SDK v5.3.1 has compatibility issues - temporarily disabled
        # TODO: Fix Deepgram SDK integration
        # if DEEPGRAM_API_KEY:
        #     stt_streamer = DeepgramStreamer(DEEPGRAM_API_KEY, on_transcript_callback)
        #     await stt_streamer.start()
        logger.info("STT (Deepgram) temporarily disabled - pending SDK fix")

        async def process_audio_buffer(ws_id):
            """Process the current audio buffer for a connection."""
            if ws_id not in self.audio_buffer or not self.audio_buffer[ws_id]:
                return
            
            logger.info("Done listening - processing buffer")
            
            # Combine all audio chunks
            current_chunks = self.audio_buffer[ws_id]
            
            # Prepend WebM header if missing (for chunks after the first buffer)
            if current_chunks and not current_chunks[0].startswith(b'\x1a\x45\xdf\xa3') and ws_id in self.session_headers:
                # Prepend initial header chunk to make this a valid WebM file
                combined_audio = self.session_headers[ws_id] + b''.join(current_chunks)
            else:
                combined_audio = b''.join(current_chunks)
            
            # Clear buffer
            self.audio_buffer[ws_id] = []
            self.is_recording[ws_id] = False
            
            # Save to temporary WebM file for transcription
            import tempfile
            temp_audio = tempfile.NamedTemporaryFile(suffix='.webm', delete=False)
            temp_wav_path = None
            try:
                # Write WebM audio data directly
                temp_audio.write(combined_audio)
                temp_audio.close()
                
                # Convert webm to wav for Deepgram compatibility
                from audio_converter import convert_webm_to_wav
                temp_wav_path = await asyncio.to_thread(convert_webm_to_wav, temp_audio.name)
                
                logger.info(f"[TRANSCRIBE] Transcribing audio from {temp_wav_path}...")
                transcribe_start = time.time()
                from speech_to_text import SpeechToText
                stt = SpeechToText()
                # Use Hinglish (Hindi + English mixed) for better understanding
                transcript = await asyncio.to_thread(stt.transcribe, temp_wav_path, language="hi,en")
                transcribe_time = time.time() - transcribe_start
                
                if transcript and transcript.strip():
                    logger.info(f"[OK] Transcription complete ({transcribe_time:.2f}s): {transcript}")
                    
                    # Send status: processing
                    await websocket.send_json({"type": "status", "status": "processing"})
                    
                    # Send user message to frontend (only once with consistent format)
                    await websocket.send_json({
                        "type": "user_message",
                        "content": transcript,
                        "message": transcript,
                        "text": transcript,
                        "timestamp": time.time(),
                        "role": "user"
                    })
                    logger.info("[OK] Sent user_message")
                    
                    # CRITICAL: Try to execute task FIRST (Action-First AI)
                    # BUT: Skip task planning for special commands like /start
                    planner = None
                    task_executed = False
                    task_result = ""
                    
                    # Skip task planning for special commands
                    if transcript.strip().lower() in ["/start", "/hello", "/hi"]:
                        logger.info(f"[SKIP] Special command detected, skipping task planner: {transcript}")
                        task_executed = False
                    else:
                        try:
                            from automation.task_planner import TaskPlanner
                            planner = TaskPlanner(llm_client=active_llm)
                            logger.info("[EXEC] Attempting to execute task...")
                            
                            # execute_plan is async, so await it directly (not with to_thread)
                            task_result = await planner.execute_plan(transcript)
                            
                            if task_result and task_result != "NO_ACTION_REQUIRED":
                                task_executed = True
                                logger.info(f"[OK] Task executed: {task_result}")
                                ai_response = task_result  # Use task result as response
                            else:
                                logger.info("[INFO] No action required, generating conversational response")
                                task_executed = False
                        except Exception as e:
                            logger.error(f"Task execution failed: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            task_executed = False
                    
                    # Generate AI response only if no task was executed
                    if not task_executed:
                        response_start = time.time()
                        logger.info(f"[AI] Generating AI response...")
                        try:
                            ai_response = await asyncio.to_thread(active_llm.generate_response, transcript)
                            response_time = time.time() - response_start
                            logger.info(f"[OK] Response generated ({response_time:.2f}s)")
                        except Exception as e:
                            logger.error(f"AI response generation failed: {e}")
                            ai_response = "I'm sorry, I'm having trouble processing that right now."
                    
                    # Send AI response to chat (only once with consistent format)
                    logger.info(f"[MSG] Sending AI response to frontend: {ai_response[:50]}...")
                    
                    # Send response
                    await websocket.send_json({
                        "type": "response",
                        "content": ai_response,
                        "message": ai_response,
                        "text": ai_response,
                        "timestamp": time.time(),
                        "role": "assistant"
                    })
                    logger.info("[OK] Sent response")
                    
                    # Send to TTS if available
                    if tts_streamer:
                        try:
                            audio_start = time.time()
                            logger.info("[AUDIO] Generating speech...")
                            await websocket.send_json({"type": "status", "status": "speaking"})
                            
                            full_audio_bytes = []
                            async for audio_chunk in tts_streamer.stream_text(ai_response):
                                full_audio_bytes.append(audio_chunk)
                                # Still send levels for animation while synthesizing
                                level = calculate_amplitude(base64.b64encode(audio_chunk).decode('utf-8'))
                                await websocket.send_json({"type": "audio_level", "payload": level})
                            
                            if full_audio_bytes:
                                final_audio = b"".join(full_audio_bytes)
                                audio_base64 = base64.b64encode(final_audio).decode('utf-8')
                                await websocket.send_json({
                                    "type": "tts_audio",
                                    "payload": audio_base64,
                                    "audio": audio_base64
                                })
                            
                            audio_time = time.time() - audio_start
                            logger.info(f"[OK] Speech generated ({audio_time:.2f}s)")
                        except Exception as e:
                            logger.error(f"TTS failed: {e}")
                            error_detail = str(e)
                            await websocket.send_json({
                                "type": "error",
                                "message": f"ElevenLabs limit reached or error: {error_detail}. Switch key in .env!"
                            })
                    
                    # Send status update
                    await websocket.send_json({"type": "status", "status": "idle"})
                else:
                    logger.info("No speech detected - sitting idle")
                    try:
                        await websocket.send_json({
                            "type": "status",
                            "status": "idle"
                        })
                    except:
                        pass
            
            except requests.exceptions.HTTPError as e:
                logger.error(f"Deepgram API error: {e}")
                if hasattr(e.response, 'text'):
                    logger.error(f"Response: {e.response.text}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Transcription failed. Please try again."
                })
                await websocket.send_json({"type": "status", "status": "idle"})
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await websocket.send_json({
                    "type": "error",
                    "message": "An error occurred during transcription."
                })
                await websocket.send_json({"type": "status", "status": "idle"})
            
            finally:
                # Clean up temp files
                import os
                try:
                    if temp_audio and os.path.exists(temp_audio.name):
                        os.unlink(temp_audio.name)
                except Exception as e:
                    logger.error(f"Error deleting temp webm file: {e}")
                try:
                    if temp_wav_path and os.path.exists(temp_wav_path) and temp_wav_path != temp_audio.name:
                        os.unlink(temp_wav_path)
                except Exception as e:
                    logger.error(f"Error deleting temp wav file: {e}")

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                msg_type = message.get("type")
                payload = message.get("payload")
                content = message.get("content")  # For simple text messages
                
                # DEBUG: Log all incoming messages
                logger.info(f"[MSG] Received message type: {msg_type}")
                
                if msg_type == "audio_chunk":
                    # Collect audio chunks and process with REST API (streaming STT disabled)
                    ws_id = str(id(websocket))
                    
                    # Initialize buffer for this connection if needed
                    if ws_id not in self.audio_buffer:
                        self.audio_buffer[ws_id] = []
                        self.buffer_start_time[ws_id] = time.time()
                        self.is_recording[ws_id] = True
                        logger.info("[MIC] Listening...")
                    
                    # Decode and store audio chunk
                    audio_bytes = base64.b64decode(payload)
                    
                    # Capture the first chunk as it contains the WebM header needed for all subsequent chunks
                    if ws_id not in self.session_headers and audio_bytes.startswith(b'\x1a\x45\xdf\xa3'):
                        # Just take the beginning of the first chunk which contains EBML/WebM headers
                        # Typically the first 1-2KB is enough, but taking the whole first chunk is safer.
                        self.session_headers[ws_id] = audio_bytes
                        logger.info("[OK] Captured WebM session header")
                    
                    self.audio_buffer[ws_id].append(audio_bytes)
                    
                    # Also calculate level for listening visual
                    level = calculate_amplitude(payload)
                    await websocket.send_json({"type": "audio_level", "payload": level})
                    
                    # Check if we have enough audio (e.g., 2 seconds worth)
                    total_size = sum(len(chunk) for chunk in self.audio_buffer[ws_id])
                    elapsed_time = time.time() - self.buffer_start_time.get(ws_id, time.time())
                    
                    # Process after 2 seconds of audio OR if buffer is large enough
                    # Explicit processing also happens on 'stop_audio' message
                    # FIX: Removed auto-processing to prevent splitting sentences. 
                    # Now waiting for explicit 'stop_audio' signal.
                    # if elapsed_time >= 2.0 or total_size >= 96000:
                    #     await process_audio_buffer(ws_id)
                
                elif msg_type == "stop_audio":
                    ws_id = str(id(websocket))
                    await process_audio_buffer(ws_id)
                
                elif msg_type == "request_metrics":
                    from system_monitor import system_monitor
                    metrics = system_monitor.get_system_info()
                    await websocket.send_json({
                        "type": "system_metrics",
                        "data": metrics
                    })
                    
                elif msg_type == "text_input" or msg_type == "message":
                    # Handle simple text messages (from chat input)
                    user_message = payload if payload else content
                    if not user_message:
                        continue
                    
                    # Send status update
                    await websocket.send_json({"type": "status", "status": "processing"})
                    
                    # Send user message to frontend (consistent with speech input)
                    await websocket.send_json({
                        "type": "user_message",
                        "content": user_message,
                        "message": user_message,
                        "text": user_message,
                        "timestamp": time.time(),
                        "role": "user"
                    })
                    
                    # Generate AI response using active LLM (Groq/Gemini)
                    try:
                        logger.info(f"Processing message: {user_message}")
                        
                        # Task Planner Integration (Action First)
                        is_executed = False
                        ai_response = ""
                        try:
                            from automation.task_planner import TaskPlanner
                            planner = TaskPlanner(llm_client=active_llm)
                            plan_result = await planner.execute_plan(user_message)
                            
                            if plan_result != "NO_ACTION_REQUIRED":
                                ai_response = plan_result
                                is_executed = True
                        except Exception as e:
                            logger.error(f"Planner error: {e}")
                            
                        if not is_executed or not ai_response:
                            ai_response = await asyncio.to_thread(active_llm.generate_response, user_message)
                        
                        # Send AI response (using frontend-expected format)
                        await websocket.send_json({
                            "type": "response",
                            "content": ai_response,
                            "message": ai_response,
                            "text": ai_response,
                            "timestamp": time.time(),
                            "role": "assistant"
                        })
                        
                        # Send to TTS if available
                        if tts_streamer:
                            try:
                                full_audio = []
                                async for audio_chunk in tts_streamer.stream_text(ai_response):
                                    full_audio.append(audio_chunk)
                                    level = calculate_amplitude(base64.b64encode(audio_chunk).decode('utf-8'))
                                    await websocket.send_json({"type": "audio_level", "payload": level})
                                
                                if full_audio:
                                    audio_base64 = base64.b64encode(b"".join(full_audio)).decode('utf-8')
                                    await websocket.send_json({
                                        "type": "tts_audio",
                                        "payload": audio_base64,
                                        "audio": audio_base64
                                    })
                            except Exception as e:
                                logger.error(f"TTS streaming failed: {e}")
                        
                        # Send status update
                        await websocket.send_json({"type": "status", "status": "idle"})
                        
                        # Log for debugging
                        logger.info(f"Sent response to client: {ai_response[:50]}...")

                    except Exception as e:
                        logger.error(f"Error generating response: {e}")
                        error_msg = "I'm sorry, I'm having trouble understanding that. Could you please rephrase?"
                        await websocket.send_json({"type": "response", "content": error_msg})
                        await websocket.send_json({"type": "error", "message": str(e)})
                        await websocket.send_json({"type": "status", "status": "idle"})

        except Exception as e:
            logger.error(f"Error in handle_voice_session: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if stt_streamer:
                try:
                    stt_streamer.stop()
                except:
                    pass  # stt_streamer might not have stop method
            self.disconnect(websocket)

    def _send_audio_to_client(self, websocket: WebSocket, base64_audio: str):
        """Helper to send audio chunk to client via callback."""
        level = calculate_amplitude(base64_audio)
        asyncio.create_task(websocket.send_json({"type": "audio_level", "payload": level}))
        asyncio.create_task(websocket.send_json({"type": "tts_audio", "payload": base64_audio}))