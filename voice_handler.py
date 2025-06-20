import asyncio
import json
import base64
import httpx
from typing import Optional
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
import config
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

class VoiceHandler:
    def __init__(self):
        """Initialize the voice handler with Deepgram client"""
        if not config.DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY not found in environment variables")
        
        # Configure Deepgram client
        self.deepgram = DeepgramClient(config.DEEPGRAM_API_KEY)
        self.api_key = config.DEEPGRAM_API_KEY
    
    async def transcribe_audio(self, audio_data: bytes, mime_type: str = "audio/wav") -> Optional[str]:
        """Transcribe audio using Deepgram's latest API"""
        try:
            # Configure transcription options
            options = {
                "smart_format": True,
                "model": "nova-2",
                "language": "en-US",
                "punctuate": True,
                "diarize": False,
                "utterances": True,
                "vad_turnoff": 500,
                "encoding": "linear16",
                "sample_rate": 16000,
                "channels": 1
            }
            
            # Send request to Deepgram
            response = await self.deepgram.transcription.prerecorded(
                {"buffer": audio_data, "mimetype": mime_type},
                options
            )
            
            # Extract transcription
            if response and "results" in response:
                transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
                return transcript.strip()
            
            return None
            
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return None
    
    async def start_live_transcription(self, callback):
        """Start live transcription with callback for results"""
        try:
            # Configure live transcription options
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                punctuate=True,
                diarize=False,
                utterances=True,
                vad_turnoff=500,
                encoding="linear16",
                sample_rate=16000,
                channels=1
            )
            
            # Set up event handlers
            self.dg_connection = self.deepgram.listen.asynclive.v("1")
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, callback)
            self.dg_connection.on(LiveTranscriptionEvents.Error, lambda e: print(f"Error: {e}"))
            self.dg_connection.on(LiveTranscriptionEvents.Close, lambda: print("Connection closed"))
            
            # Start the connection
            await self.dg_connection.start(options)
            
        except Exception as e:
            print(f"Error starting live transcription: {str(e)}")
            raise
    
    async def stop_live_transcription(self):
        """Stop live transcription"""
        try:
            if hasattr(self, 'dg_connection'):
                await self.dg_connection.finish()
        except Exception as e:
            print(f"Error stopping live transcription: {str(e)}")
    
    async def synthesize_speech_browser(self, text: str) -> dict:
        """Generate speech using Deepgram's TTS API"""
        try:
            async with httpx.AsyncClient() as client:
                tts_response = await client.post(
                    "https://api.deepgram.com/v1/speak",
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Accept": "audio/mulaw"
                    },
                    params={
                        "model": "aura-asteria-en",
                        "encoding": "mulaw",
                        "sample_rate": 16000,
                        "speed": 1.2
                    },
                    json={
                        "text": text
                    }
                )
                
                if tts_response.status_code == 200:
                    # Convert the audio content to base64
                    audio_content = base64.b64encode(tts_response.content).decode('utf-8')
                    
                    return {
                        "type": "tts",
                        "text": text,
                        "audio": audio_content,
                        "format": "audio/mulaw",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    print(f"Deepgram TTS API error: {tts_response.status_code} - {tts_response.text}")
                    # Fallback to browser TTS
                    return {
                        "type": "tts_browser",
                        "text": text,
                        "timestamp": datetime.now().isoformat()
                    }
            
        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")
            # Fallback to browser TTS
            return {
                "type": "tts_browser",
                "text": text,
                "timestamp": datetime.now().isoformat()
            }

# Singleton instance
_voice_handler_instance = None

def get_voice_handler() -> VoiceHandler:
    """Get or create the voice handler instance"""
    global _voice_handler_instance
    if _voice_handler_instance is None:
        _voice_handler_instance = VoiceHandler()
    return _voice_handler_instance