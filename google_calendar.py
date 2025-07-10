import datetime
from typing import List, Tuple
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os
import pickle
import pytz
import io
import queue
import re
import time
import json

from dotenv import load_dotenv

import pyaudio
import pygame
import speech_recognition as sr
import httpx

import google.generativeai as genai

import config

load_dotenv()

# Constants
SCOPES = ['https://www.googleapis.com/auth/calendar']
IST = pytz.timezone('Asia/Kolkata')
WORK_START_HOUR = 9
WORK_END_HOUR = 19    

# --- API Keys and Clients ---
DEEPGRAM_API_KEY = config.DEEPGRAM_API_KEY

# --- Audio Recording Parameters ---
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

# Initialize audio playback (Pygame for playing TTS audio)
# Pygame mixer needs to be initialized only once
pygame.mixer.init()

class GoogleCalendarService:
    def __init__(self):
        self.service = self._authenticate()

    def _authenticate(self):
        creds = None
        if os.path.exists('token.json'):
            try:
                with open('token.json', 'rb') as token:
                    creds = pickle.load(token)
            except pickle.UnpicklingError:
                print("Warning: token.json is corrupted. Deleting and re-authenticating.")
                os.remove('token.json')
                creds = None # Force re-authentication

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                # Use a fixed port and explicitly set the redirect URI
                creds = flow.run_local_server(
                    port=5000,
                    authorization_prompt_message='Please visit this URL: {url}',
                    success_message='The auth flow is complete; you may close this window.',
                    open_browser=True
                )

            with open('token.json', 'wb') as token:
                pickle.dump(creds, token)

        return build('calendar', 'v3', credentials=creds)

    def get_busy_events_for_day(self, start_time: datetime.datetime, end_time: datetime.datetime) -> List[Tuple[datetime.datetime, datetime.datetime]]:
        time_min = start_time.astimezone(pytz.UTC).isoformat()
        time_max = end_time.astimezone(pytz.UTC).isoformat()

        events_result = self.service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])
        
        busy_slots = []
        for event in events:
            start = event['start'].get('dateTime')
            end = event['end'].get('dateTime')

            if start and end:
                start_dt_utc = datetime.datetime.fromisoformat(start.replace('Z', '+00:00'))
                end_dt_utc = datetime.datetime.fromisoformat(end.replace('Z', '+00:00'))

                busy_slots.append((
                    start_dt_utc.astimezone(IST),
                    end_dt_utc.astimezone(IST)
                ))
        return busy_slots

    def book_meeting(self, start_time: datetime.datetime, end_time: datetime.datetime, summary: str = "Appointment", phone_number: str = None) -> str:
        event = {
            'summary': summary,
            'start': {
                'dateTime': start_time.astimezone(pytz.UTC).isoformat(),
                'timeZone': 'Asia/Kolkata'
            },
            'end': {
                'dateTime': end_time.astimezone(pytz.UTC).isoformat(),
                'timeZone': 'Asia/Kolkata'
            },
            'reminders': {
                'useDefault': True,
            }
        }
        if phone_number:
            event['description'] = f"Phone Number: {phone_number}"

        created_event = self.service.events().insert(calendarId='primary', body=event).execute()
        return created_event.get('htmlLink', '')

    def get_events_by_phone_number(self, phone_number: str) -> List[dict]:
        # Search for events in a reasonable time range (e.g., 1 year in the past, 1 year in the future)
        now = datetime.datetime.now(pytz.UTC)
        time_min = (now - datetime.timedelta(days=365)).isoformat()
        time_max = (now + datetime.timedelta(days=365)).isoformat()

        events_result = self.service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])

        matching_events = []
        for event in events:
            description = event.get('description', '')
            if f"Phone Number: {phone_number}" in description:
                matching_events.append(event)
        return matching_events

_calendar_service_instance = None

def get_calendar_service_instance():
    global _calendar_service_instance
    if _calendar_service_instance is None:
        _calendar_service_instance = GoogleCalendarService()
    return _calendar_service_instance

def get_busy_events_for_day(start_time: datetime.datetime, end_time: datetime.datetime) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    service_instance = get_calendar_service_instance()
    return service_instance.get_busy_events_for_day(start_time, end_time)

def find_free_slots(start_time: datetime.datetime, end_time: datetime.datetime, duration_minutes: int = 60) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Find all available time slots for a meeting of the specified duration within a given time range.
    
    Args:
        start_time: Start of the time range (datetime object in IST).
        end_time: End of the time range (datetime object in IST).
        duration_minutes: Duration of the meeting in minutes.
        
    Returns:
        List of (start_time, end_time) tuples in the local timezone (IST)
    """
    busy_slots = get_busy_events_for_day(start_time, end_time)
    busy_slots.sort()
    
    # Calculate slot duration
    slot_duration = datetime.timedelta(minutes=duration_minutes)
    
    # Filter out slots that overlap with busy times
    free_slots = []
    current_time = start_time

    for busy_start, busy_end in busy_slots:
        # Add free slots before the current busy slot, within the requested range
        while current_time + slot_duration <= busy_start and current_time + slot_duration <= end_time:
            free_slots.append((current_time, current_time + slot_duration))
            current_time += slot_duration # Move to next interval by slot_duration
        
        # Move current_time past the busy slot
        current_time = max(current_time, busy_end)

    # Add free slots after the last busy slot until the end of the requested range
    while current_time + slot_duration <= end_time:
        free_slots.append((current_time, current_time + slot_duration))
        current_time += slot_duration # Move to next interval by slot_duration
    
    # Remove duplicates and sort - this step is still necessary if intervals overlap or due to logic
    free_slots = sorted(list(set(free_slots)))
    
    return free_slots

def format_slots(slots: List[Tuple[datetime.datetime, datetime.datetime]]) -> List[str]:
    return [f"{start.strftime('%I:%M %p')} - {end.strftime('%I:%M %p')}" for start, end in slots]

def book_meeting(start_time: datetime.datetime, end_time: datetime.datetime, summary: str = "Appointment", phone_number: str = None) -> str:
    service_instance = get_calendar_service_instance()
    return service_instance.book_meeting(start_time, end_time, summary, phone_number)

def update_appointment(event_id: str, new_start_time: datetime.datetime, new_end_time: datetime.datetime, new_summary: str = "Appointment", phone_number: str = None) -> str:
    service_instance = get_calendar_service_instance()

    updated_event_body = {
        'summary': new_summary,
        'start': {
            'dateTime': new_start_time.astimezone(pytz.UTC).isoformat(),
            'timeZone': 'Asia/Kolkata'
        },
        'end': {
            'dateTime': new_end_time.astimezone(pytz.UTC).isoformat(),
            'timeZone': 'Asia/Kolkata'
        },
        'reminders': {
            'useDefault': True,
        }
    }
    if phone_number:
        updated_event_body['description'] = f"Phone Number: {phone_number}"

    try:
        service_instance.service.events().update(
            calendarId='primary',
            eventId=event_id,
            body=updated_event_body
        ).execute()
        return "Appointment updated successfully."
    except Exception as e:
        return f"Error updating appointment: {e}"

def delete_appointment(phone_number: str, summary: str = None) -> str:
    service_instance = get_calendar_service_instance()
    matching_events = service_instance.get_events_by_phone_number(phone_number)

    if not matching_events:
        return "No appointments found for the given phone number."

    # Filter by summary if provided
    if summary:
        lower_summary = summary.lower()
        filtered_events = [event for event in matching_events if lower_summary in event.get('summary', '').lower()]
    else:
        filtered_events = matching_events

    if not filtered_events:
        return "No appointments found with that phone number and matching title."

    deleted_count = 0
    for event in filtered_events:
        event_id = event['id']
        try:
            service_instance.service.events().delete(
                calendarId='primary',
                eventId=event_id
            ).execute()
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting event {event_id}: {e}")
            # Continue to next event if one fails

    return f"Successfully deleted {deleted_count} appointment(s)."

# --- LLM Interaction Function ---
def get_gemini_response(chat: genai.GenerativeModel.start_chat, prompt: str) -> str:
    """
    Gets a response from the Gemini LLM for a given prompt using a persistent chat session.
    
    Args:
        chat: The persistent Gemini chat object.
        prompt: The text prompt to send to the LLM.
        
    Returns:
        The LLM's text response, or a simplified error message.
    """
    try:
        print(f"[DEBUG] Sending to Gemini: {prompt}") # Log what's sent to Gemini
        response = chat.send_message(prompt, 
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 2048
            }
        )
        
        # Extract text response, assuming it's a simple text response
        return response.text.strip() if response.text else "I'm sorry, I couldn't generate a text response."
    except Exception as e:
        error_message = str(e)
        print(f"An error occurred with the LLM: {error_message}") # Log raw error
        if "quota" in error_message.lower() or "429" in error_message:
            # Return a very short, speakable message for TTS
            return "My apologies, I've hit my usage limit. Please wait a moment."
        else:
            # For other errors, return a generic, speakable message
            return "I encountered an unexpected error. Please try again."

# --- Speech-to-Text (STT) Function (Synchronous) ---
def listen_for_speech_sync(device_index: int = 2) -> str:
    """
    Listens for speech from the microphone and transcribes it using Google Web Speech API (synchronously).
    
    Args:
        device_index: The index of the input audio device to use.
        
    Returns:
        The transcribed text, or an empty string if no speech is recognized.
    """
    recognizer = sr.Recognizer()
    # Adjust for ambient noise for better accuracy
    recognizer.energy_threshold = 3000  # Lower threshold for better sensitivity
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8  # Slightly longer pause before considering the end of speech
    recognizer.operation_timeout = 10  # Timeout for operations (max 10s for recognition)

    print("\nListening for your input... (Speak now)")
    with sr.Microphone(device_index=device_index, sample_rate=RATE) as source:
        print("Adjusting for ambient noise, please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Microphone adjusted. Speak now.")
        try:
            # Listen for up to 5 seconds of speech, with a phrase limit of 10 seconds
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10) 
            print("Audio captured. Transcribing...")
            
            # Use Google Web Speech API for transcription
            text = recognizer.recognize_google(audio)
            print(f"User: {text}")
            return text.strip()
        except sr.WaitTimeoutError:
            print("No speech detected after initial wait.")
            return ""
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")
            return ""
        except Exception as e:
            print(f"An unexpected error occurred during speech recognition: {e}")
            return ""

# --- Text-to-Speech (TTS) Function (Synchronous) ---
def speak_text_sync(text: str):
    """
    Synthesizes speech using Deepgram TTS (synchronously) and plays it.
    """
    if not text or not text.strip(): # Ensure text is not empty or just whitespace
        print("TTS: Received empty text, skipping speech synthesis.")
        return

    # Print the text being spoken *before* the API call
    # This addresses the user's request to see what's being sent to TTS
    print(f"AI: {text}") 
    print(f"[DEBUG] Type of text: {type(text)}, Raw repr: {repr(text)}") # Debug print raw text

    url = "https://api.deepgram.com/v1/speak"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg", # Explicitly request MP3 audio
    }
    payload = {
        "text": text,
        "model": "aura-asteria-en", # A good default voice
        "encoding": "linear16", # Raw audio format
        "sample_rate": RATE,
    }

    try:
        # Use httpx.post with json=payload for correct JSON serialization
        response = httpx.post(url, headers=headers, json=payload, timeout=10.0) 
        
        if response.status_code == 200:
            audio_data = response.content
            
            # Play audio using Pygame
            audio_stream = io.BytesIO(audio_data)
            audio_stream.seek(0)
            
            pygame.mixer.music.load(audio_stream, "mp3") # Load as MP3, Pygame can handle BytesIO
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1) # Wait for playback to finish
            pygame.mixer.music.stop() # Ensure it's stopped after playing

        else:
            error_text = response.text
            print(f"Deepgram TTS API error: {response.status_code} - {error_text}")
            print(f"TTS Error Details (Payload Sent): {json.dumps(payload)}") 
            # Raw Request Content Sent is not directly available from response.request.content in httpx for sync post
            # if response.request and hasattr(response.request, 'content'):
            #     print(f"Raw Request Content Sent: {response.request.content.decode('utf-8', errors='ignore')}")

    except Exception as e:
        print(f"Error in Deepgram TTS request/playback: {e}", flush=True)

# --- Main Application Loop (Synchronous) ---
def main():
    print("🎙️ Start speaking. Say 'exit' to quit.\n")

    # Initial check for Deepgram API key
    if not DEEPGRAM_API_KEY:
        print("Error: DEEPGRAM_API_KEY not found in config.py or environment variables.")
        print("Please set it up before running the agent.")
        return

    print("AI agent ready! Speak to interact or press Ctrl+C to quit.")
    
    # Initialize LLM model and chat session once
    try:
        model = get_llm()
        chat = model.start_chat(history=[])
        print("Gemini LLM chat session initialized.")
    except Exception as e:
        print(f"Error initializing Gemini LLM: {e}")
        print("Cannot proceed without a working LLM. Exiting.")
        return

    try:
        while True:
            # 1. Listen for speech (STT) synchronously
            # Pass only device_index, as Deepgram API key is not needed for recognize_google
            user_input = listen_for_speech_sync(device_index=2) 
            
            if not user_input:
                print("Trying again...\n")
                continue
                
            # user_input is already printed by listen_for_speech_sync
            
            # Check for exit command
            if re.search(r"\b(exit|quit)\b", user_input, re.I):
                speak_text_sync("Goodbye!")
                print("👋 Exiting.")
                break
            
            # 2. Get response from Gemini LLM
            print("AI Thinking...")
            # Pass the persistent chat object
            gemini_response = get_gemini_response(chat, user_input) 
            
            # 3. Speak the AI's response (TTS) synchronously
            # This function now also prints the AI's response before attempting TTS
            speak_text_sync(gemini_response)
            
            time.sleep(1) # Small pause before next listening round
            
    except KeyboardInterrupt:
        print("\n\nApplication terminated by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup pygame mixer
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        print("\nThank you for using the AI agent. Goodbye!")

if __name__ == "__main__":
    main()