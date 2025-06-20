from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import logging
import asyncio
from typing import Dict, Any
import uvicorn
import os
import re
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from graph_agent import get_agent
from voice_handler import get_voice_handler
import config

# OAuth 2.0 Configuration for Google Calendar
# NOTE: These should ideally come from a secure configuration management or environment variables
CLIENT_CONFIG = {
    "web": {
        "client_id": config.GOOGLE_CLIENT_ID,
        "client_secret": config.GOOGLE_CLIENT_SECRET,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "redirect_uris": ["http://localhost:5000/oauth2callback"],
        "javascript_origins": ["http://localhost:5000"]
    }
}
SCOPES = config.SCOPES
REDIRECT_URI = "http://localhost:5000/oauth2callback" # Ensure this matches a redirect URI in your Google Cloud Console

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Scheduler AI Agent")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize agent and voice handler
agent = get_agent()
voice_handler = get_voice_handler()

# Store conversation contexts per session
conversation_contexts: Dict[str, Dict[str, Any]] = {}

@app.get("/")
async def index(request: Request):
    """Serve the main HTML interface or initiate OAuth"""
    token_path = 'token.json'
    if not os.path.exists(token_path) or os.path.getsize(token_path) == 0:
        logger.info("token.json not found or empty. Redirecting to OAuth.")
        return RedirectResponse(url="/auth")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/auth")
async def auth_redirect():
    """Initiate the Google OAuth2 flow"""
    try:
        flow = Flow.from_client_config(
            CLIENT_CONFIG,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='select_account consent' # Force account selection and consent
        )
        
        logger.info(f"Redirecting to authorization URL: {authorization_url}")
        return RedirectResponse(authorization_url)
    except Exception as e:
        logger.error(f"Error in auth endpoint: {str(e)}")
        return JSONResponse({
            'success': False,
            'error': f"Error starting authentication flow: {str(e)}"
        })

@app.get("/oauth2callback")
async def oauth2callback(request: Request):
    """Handle the OAuth2 callback from Google"""
    try:
        # Get the authorization code from the request
        code = request.query_params.get('code')
        if not code:
            logger.error("No authorization code provided in callback.")
            raise HTTPException(status_code=400, detail="No authorization code provided")
        
        # Create a new flow for this callback
        flow = Flow.from_client_config(
            CLIENT_CONFIG,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        
        # Exchange the code for credentials
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        # Save the credentials
        token_path = 'token.json'
        with open(token_path, 'w') as token_file:
            token_file.write(credentials.to_json())
        logger.info(f"Credentials saved successfully to {token_path}")
        
        # Attempt to re-authenticate the calendar tool now that token.json is available
        agent.ensure_calendar_authenticated()
        
        # Redirect to the main page or a success page
        return RedirectResponse(url="/")
    except Exception as e:
        logger.error(f"Error in oauth2callback: {str(e)}", exc_info=True)
        # If there's an error, redirect back to auth to start fresh
        return RedirectResponse(url="/auth")

@app.post("/api/chat")
async def chat(request: Request):
    """Handle chat messages"""
    try:
        data = await request.json()
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        logger.info(f"Received chat message: {message}")
        
        if not message:
            return JSONResponse({
                'success': False,
                'error': "No message provided"
            })
        
        # Get conversation history for this session
        conversation_history = conversation_contexts.get(session_id, {}).get('history', [])
        
        # Check if a booking just occurred and the user is confirming it
        current_context = conversation_contexts.get(session_id, {}).get('context', {})
        user_input_lower = message.lower()
        booking_confirmation_phrases = ['yes', 'sure', 'ok', 'book it', 'confirm']
        
        # Determine if user wants to book a specific slot (even if not explicitly confirmed yet)
        explicit_slot_selection = None
        slot_match = re.search(r'(?:option\s*|book\s*option\s*|book\s*)(\d+)', user_input_lower)
        if slot_match:
            try:
                explicit_slot_selection = int(slot_match.group(1)) - 1 # Convert to 0-indexed
                logger.info(f"Detected explicit slot selection: {explicit_slot_selection}")
            except ValueError:
                pass # Not a valid number

        if (current_context.get('last_action') == 'meeting_booked' and
            any(phrase in user_input_lower for phrase in booking_confirmation_phrases)):
            logger.info(f"Skipping agent processing for confirmed booking message: {message}")
            # Reset context here as well to ensure consistent state
            conversation_contexts[session_id] = {
                'context': {
                    'step': 'initial',
                    'duration': None,
                    'preferred_time': None,
                    'title': None,
                    'available_slots': None,
                    'last_action': None # Clear last_action after acknowledging
                },
                'history': []
            }
            return JSONResponse({
                'success': True,
                'response': "Your meeting has been successfully booked.",
                'context': conversation_contexts[session_id]['context'],
                'has_slots': False,
                'available_slots': [],
                'slot_index': None,
                'slot_time': None
            })

        # Process message with agent, passing the selected slot index if found
        result = agent.process_message(message, conversation_history, selected_slot_index=explicit_slot_selection)
        logger.info(f"Agent response: {result}")
        
        if not result or not result.get('success'):
            return JSONResponse({
                'success': False,
                'error': result.get('error', "I'm sorry, I couldn't process that request.")
            })
        
        # Update conversation context for this session
        if session_id not in conversation_contexts:
            conversation_contexts[session_id] = {}
        
        conversation_contexts[session_id]['context'] = result.get('conversation_context', {})
        conversation_contexts[session_id]['history'] = result.get('conversation_history', conversation_history)
        
        # Check if we have available slots
        available_slots = result.get('available_slots', [])
        has_slots = result.get('has_slots', False)
        
        # Determine if user wants to book a specific slot (for frontend display)
        slot_index_for_frontend = None
        if has_slots and available_slots:
            # Check if user input suggests booking a specific slot
            user_input_lower = message.lower()
            if any(word in user_input_lower for word in ['yes', 'sure', 'ok', 'book', 'confirm', 'first', 'second', 'third']):
                # User wants to book - determine which slot
                if 'first' in user_input_lower or '1' in user_input_lower:
                    slot_index_for_frontend = 0
                elif 'second' in user_input_lower or '2' in user_input_lower:
                    slot_index_for_frontend = 1
                elif 'third' in user_input_lower or '3' in user_input_lower:
                    slot_index_for_frontend = 2
                else:
                    # Default to first slot
                    slot_index_for_frontend = 0
        
        return JSONResponse({
            'success': True,
            'response': result.get('response', "I'm sorry, I couldn't process that request."),
            'context': result.get('conversation_context', {}),
            'has_slots': has_slots,
            'available_slots': available_slots,
            'slot_index': slot_index_for_frontend,
            'slot_time': None  # Will be set when booking
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return JSONResponse({
            'success': False,
            'error': f"I'm sorry, I encountered an error: {str(e)}"
        })

@app.post("/api/book")
async def book_meeting(request: Request):
    """Book a meeting for a specific slot"""
    try:
        data = await request.json()
        slot_index = data.get('slot_index')
        title = data.get('title', 'Meeting')
        attendees = data.get('attendees', [])
        session_id = data.get('session_id', 'default')
        
        # Get conversation context for this session
        context = conversation_contexts.get(session_id, {}).get('context', {})
        available_slots = context.get('available_slots', [])
        
        if slot_index is None or slot_index >= len(available_slots):
            return JSONResponse({
                'success': False,
                'error': "Invalid slot selection"
            })
        
        slot = available_slots[slot_index]
        
        # Create meeting using the calendar tool
        try:
            result = agent.calendar_tool.invoke(
                action="create_meeting",
                start_time=slot["start_time"],
                end_time=slot["end_time"],
                title=title
            )
            
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {"error": f"Invalid response: {result}"}
            
            if result.get("success"):
                # Reset conversation context after successful booking
                conversation_contexts[session_id] = {
                    'context': {
                        'step': 'initial',
                        'duration': None,
                        'preferred_time': None,
                        'title': None,
                        'available_slots': None,
                        'last_action': 'meeting_booked'
                    },
                    'history': []
                }
                
                return JSONResponse({
                    'success': True,
                    'message': f"Perfect! I've successfully booked your meeting. {result.get('message', '')} You can view it at: {result.get('event_link', '')}"
                })
            else:
                return JSONResponse({
                    'success': False,
                    'error': result.get('error', "I'm sorry, I couldn't book that slot. Please try again.")
                })
                
        except Exception as e:
            logger.error(f"Error creating meeting: {str(e)}")
            return JSONResponse({
                'success': False,
                'error': f"Error creating meeting: {str(e)}"
            })
            
    except Exception as e:
        logger.error(f"Error in book_meeting endpoint: {str(e)}")
        return JSONResponse({
            'success': False,
            'error': "I'm sorry, I couldn't book that slot. Please try again."
        })

@app.post("/api/reset")
async def reset_conversation(request: Request):
    """Reset conversation context"""
    try:
        data = await request.json()
        session_id = data.get('session_id', 'default')
        
        conversation_contexts[session_id] = {
            'context': {
                'step': 'initial',
                'duration': None,
                'preferred_time': None,
                'title': None,
                'available_slots': None,
                'last_action': None
            },
            'history': []
        }
        
        return JSONResponse({'success': True})
    except Exception as e:
        logger.error(f"Error in reset_conversation endpoint: {str(e)}")
        return JSONResponse({
            'success': False,
            'error': "Error resetting conversation"
        })

@app.post("/api/text-to-speech")
async def text_to_speech(request: Request):
    """Convert text to speech"""
    try:
        data = await request.json()
        text = data.get('text', '').strip()
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No text provided"}
            )
        
        # Use voice handler to synthesize speech
        tts_data = await voice_handler.synthesize_speech_browser(text)
        
        if tts_data.get("type") == "tts_browser":
            # Return browser TTS data
            return JSONResponse({
                "success": True,
                "type": "tts_browser",
                "text": text
            })
        elif tts_data.get("type") == "tts":
            # Return the audio data
            from fastapi.responses import Response
            import base64
            
            audio_data = base64.b64decode(tts_data["audio"])
            
            return Response(
                content=audio_data,
                media_type=tts_data["format"],
                headers={"Content-Disposition": "attachment; filename=speech.mulaw"}
            )
        else:
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": "TTS service unavailable"}
            )
        
    except Exception as e:
        logger.error(f"Error in text_to_speech endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"TTS error: {str(e)}"}
        )

@app.post("/api/speech/recognize")
async def recognize_speech(request: Request):
    """Recognize speech from audio"""
    try:
        from fastapi import UploadFile, File
        import base64
        
        # Get the audio data from the request
        form = await request.form()
        audio_file = form.get('audio')
        
        if not audio_file:
            return JSONResponse({
                'success': False,
                'error': "No audio file provided"
            })
        
        # Read the audio data
        audio_data = await audio_file.read()
        
        # Transcribe using voice handler
        transcription = await voice_handler.transcribe_audio(audio_data, audio_file.content_type)
        
        if not transcription:
            return JSONResponse({
                'success': False,
                'error': "Could not transcribe audio. Please try again."
            })
        
        # Process the transcription with the agent
        session_id = 'default'  # You might want to get this from the request
        conversation_history = conversation_contexts.get(session_id, {}).get('history', [])
        
        result = agent.process_message(transcription, conversation_history)
        
        # Update conversation context
        if session_id not in conversation_contexts:
            conversation_contexts[session_id] = {}
        
        conversation_contexts[session_id]['context'] = result.get('conversation_context', {})
        conversation_contexts[session_id]['history'] = result.get('conversation_history', conversation_history)
        
        return JSONResponse({
            'success': True,
            'text': transcription,
            'response': result.get('response', ''),
            'context': result.get('conversation_context', {}),
            'has_slots': result.get('has_slots', False),
            'available_slots': result.get('available_slots', [])
        })
        
    except Exception as e:
        logger.error(f"Error in speech recognition: {str(e)}")
        return JSONResponse({
            'success': False,
            'error': f"Speech recognition error: {str(e)}"
        })

@app.get("/api/calendar/today")
async def get_today_meetings():
    """Get today's meetings"""
    try:
        if not agent.calendar_tool:
            return JSONResponse({
                'success': False,
                'error': "Calendar service not available"
            })
        
        # Get today's meetings using the calendar tool
        result = agent.calendar_tool.invoke(action="get_today_meetings")
        
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = {"error": f"Invalid response: {result}"}
        
        if result.get("success"):
            return JSONResponse({
                'success': True,
                'meetings': result.get('meetings', [])
            })
        else:
            return JSONResponse({
                'success': False,
                'error': result.get('error', "Could not fetch today's meetings")
            })
            
    except Exception as e:
        logger.error(f"Error getting today's meetings: {str(e)}")
        return JSONResponse({
            'success': False,
            'error': f"Error fetching meetings: {str(e)}"
        })

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level="info"
    ) 