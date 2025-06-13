import os
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import logging
import base64
import speech_recognition as sr
from functools import lru_cache

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import dateutil.parser
from dateutil.relativedelta import relativedelta
import dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize speech recognizer
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300  # Lower threshold for better sensitivity
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.5  # Shorter pause threshold for faster response

# Google Calendar API scope
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly', 
          'https://www.googleapis.com/auth/calendar.events']

# Add after other global variables
SLOT_CACHE_DURATION = 300  # 5 minutes in seconds
last_slot_check = None
cached_slots = None

class SmartSchedulerAgent:
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.eleven_labs_api_key = os.getenv('ELEVEN_LABS_API_KEY')
        self.calendar_service = None
        self.conversation_context = {
            'duration': None,
            'preferred_time': None,
            'title': None,
            'attendees': [],
            'current_step': 'initial',
            'available_slots': []
        }
        
        # Initialize Gemini AI
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            logger.warning("GEMINI_API_KEY not found. AI responses will be simulated.")
            self.model = None
        
        # Initialize Eleven Labs
        if self.eleven_labs_api_key:
            self.elevenlabs_client = ElevenLabs(api_key=self.eleven_labs_api_key)
            self.elevenlabs_voice_id = 'EXAVITQu4vr4xnSDxMaL' # A common default voice ID, change if you have a preferred one
            logger.info("Eleven Labs API initialized successfully")
        else:
            logger.warning("ELEVEN_LABS_API_KEY not found. Eleven Labs features will be disabled.")
            self.elevenlabs_client = None
            self.elevenlabs_voice_id = None

        # Initialize Google Calendar
        self.setup_calendar_service()
    
    def setup_calendar_service(self):
        """Initialize Google Calendar service"""
        creds = None
        
        # Try to load existing credentials
        if os.path.exists('token.json'):
            try:
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
                logger.info("Loaded existing credentials from token.json")
                
                # Check if credentials are valid and not expired
                if creds and creds.valid:
                    logger.info("Existing credentials are valid")
                    self.calendar_service = build('calendar', 'v3', credentials=creds)
                    return True
                elif creds and creds.expired and creds.refresh_token:
                    logger.info("Refreshing expired credentials")
                    creds.refresh(Request())
                    # Save the refreshed credentials
                    with open('token.json', 'w') as token:
                        token.write(creds.to_json())
                    self.calendar_service = build('calendar', 'v3', credentials=creds)
                    return True
            except Exception as e:
                logger.error(f"Error loading existing credentials: {e}")
                # If there's an error loading credentials, we'll start a new OAuth flow
                if os.path.exists('token.json'):
                    try:
                        os.remove('token.json')
                        logger.info("Removed invalid token.json")
                    except Exception as e:
                        logger.error(f"Error removing token.json: {e}")

        # Get new credentials if needed
        try:
            # Use client ID, client secret, and redirect URI from environment variables
            client_id = os.getenv('GOOGLE_CLIENT_ID')
            client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
            redirect_uri = os.getenv('GOOGLE_REDIRECT_URI')

            if not all([client_id, client_secret, redirect_uri]):
                logger.warning("Google Client ID, Client Secret, or Redirect URI not found in .env. Calendar features will be simulated.")
                return False

            logger.info("Starting new OAuth flow...")
            flow = InstalledAppFlow.from_client_config(
                {
                    "web": {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [redirect_uri]
                    }
                }, SCOPES)
            
            # Start the OAuth flow with a specific port and prompt
            creds = flow.run_local_server(
                port=5000,
                prompt='consent',  # Force consent screen
                authorization_prompt_message='Please authorize the application to access your Google Calendar.',
                success_message='Authentication successful! Redirecting to home page...',
                open_browser=True
            )
            
            logger.info("OAuth flow completed successfully")
            
            # Save credentials for next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
                logger.info("Saved new credentials to token.json")
            
            # Build the calendar service
            self.calendar_service = build('calendar', 'v3', credentials=creds)
            logger.info("Google Calendar service initialized successfully")
            
            return True
        
        except Exception as e:
            logger.error(f"Error during OAuth flow: {e}")
            return False
    
    def parse_time_with_ai(self, user_input: str) -> Dict:
        """Use AI to parse complex time expressions"""
        if not self.model:
            logger.info("Using fallback time parsing (no AI model available)")
            return self._fallback_time_parsing(user_input)
        
        prompt = f"""
        Parse this scheduling request and extract the key information:
        "{user_input}"
        
        Return a JSON object with these fields:
        - duration: meeting duration (e.g., "30 minutes", "1 hour", "2 hours")
        - preferred_time: general time preference (e.g., "morning", "afternoon", "evening")
        - specific_date: specific date if mentioned (e.g., "2024-06-20", "tomorrow")
        - urgency: how urgent (e.g., "asap", "flexible", "specific")
        - meeting_type: type of meeting if mentioned
        
        Only include fields that are clearly mentioned. Return valid JSON only.
        """
        
        try:
            logger.info("Sending request to Gemini AI")
            response = self.model.generate_content(prompt)
            logger.info(f"Raw AI response: {response.text}")
            
            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            
            # Remove markdown code block formatting if present
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            
            # Ensure the response starts and ends with curly braces
            if not response_text.startswith('{'):
                response_text = '{' + response_text
            if not response_text.endswith('}'):
                response_text = response_text + '}'
            
            # Clean up any trailing commas before closing braces
            response_text = re.sub(r',\s*}', '}', response_text)
            
            parsed_data = json.loads(response_text)
            logger.info(f"Parsed AI response: {parsed_data}")
            return parsed_data
        except Exception as e:
            logger.error(f"AI parsing failed: {str(e)}")
            logger.info("Falling back to basic time parsing")
            return self._fallback_time_parsing(user_input)
    
    def _fallback_time_parsing(self, user_input: str) -> Dict:
        """Fallback time parsing without AI"""
        logger.info("Using fallback time parsing")
        result = {}
        text = user_input.lower()
        
        # Parse duration
        duration_patterns = [
            (r'(\d+)\s*hours?', lambda m: f"{m.group(1)} hour{'s' if int(m.group(1)) > 1 else ''}"),
            (r'(\d+)\s*hrs?', lambda m: f"{m.group(1)} hour{'s' if int(m.group(1)) > 1 else ''}"),
            (r'(\d+)\s*minutes?', lambda m: f"{m.group(1)} minutes"),
            (r'(\d+)\s*mins?', lambda m: f"{m.group(1)} minutes"),
        ]
        
        for pattern, formatter in duration_patterns:
            match = re.search(pattern, text)
            if match:
                result['duration'] = formatter(match)
                logger.info(f"Found duration: {result['duration']}")
                break
        
        # Parse time preferences
        if 'morning' in text:
            result['preferred_time'] = 'morning'
        elif 'afternoon' in text:
            result['preferred_time'] = 'afternoon'
        elif 'evening' in text:
            result['preferred_time'] = 'evening'
        elif 'tomorrow' in text:
            result['specific_date'] = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        elif 'next week' in text:
            result['preferred_time'] = 'next week'
        
        # Parse urgency
        if 'asap' in text or 'urgent' in text or 'immediately' in text:
            result['urgency'] = 'asap'
        elif 'flexible' in text:
            result['urgency'] = 'flexible'
        
        logger.info(f"Fallback parsing result: {result}")
        return result
    
    @lru_cache(maxsize=32)
    def find_available_slots(self, duration_minutes: int = 60, 
                           preferred_time: str = None, 
                           days_ahead: int = 7) -> List[Dict]:
        """Find available time slots in the calendar with caching"""
        logger.info(f"Finding available slots for {duration_minutes} minutes")
        
        if not self.calendar_service:
            return self._simulate_available_slots(duration_minutes, preferred_time)
        
        try:
            # Use IST for all time calculations
            ist = timezone(timedelta(hours=5, minutes=30))
            now_ist = datetime.now(ist)
            
            target_dates = []
            specific_date_str = self.conversation_context.get('specific_date')
            
            if specific_date_str:
                try:
                    parsed_date = dateutil.parser.parse(specific_date_str, fuzzy=True)
                    # Ensure the date is in IST
                    if parsed_date.tzinfo is None:
                        parsed_date = parsed_date.replace(tzinfo=ist)
                    else:
                        parsed_date = parsed_date.astimezone(ist)

                    if parsed_date.date() < now_ist.date():
                        parsed_date += timedelta(weeks=1)
                        if parsed_date.tzinfo is None:
                            parsed_date = parsed_date.replace(tzinfo=ist)

                    target_dates.append(parsed_date.replace(hour=0, minute=0, second=0, microsecond=0))
                    logger.info(f"Specific date parsed: {parsed_date.date()}")
                except Exception as e:
                    logger.warning(f"Could not parse specific_date '{specific_date_str}': {e}. Falling back to days_ahead.")
                    for i in range(days_ahead):
                        day = now_ist + timedelta(days=i)
                        target_dates.append(day.replace(hour=0, minute=0, second=0, microsecond=0))
            else:
                for i in range(days_ahead):
                    day = now_ist + timedelta(days=i)
                    target_dates.append(day.replace(hour=0, minute=0, second=0, microsecond=0))

            available_slots = []
            for day_start_ist in target_dates:
                # Convert IST times to UTC for Google Calendar API
                day_start_utc = day_start_ist.astimezone(timezone.utc)
                day_end_utc = (day_start_ist + timedelta(days=1) - timedelta(seconds=1)).astimezone(timezone.utc)

                time_min = day_start_utc.isoformat()
                time_max = day_end_utc.isoformat()

                logger.info(f"Checking calendar for slots between {time_min} and {time_max}")

                events_result = self.calendar_service.events().list(
                    calendarId='primary',
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute()
                events = events_result.get('items', [])
                
                # Create initial free slot for the day (e.g., 9 AM to 5 PM) in IST
                day_start_time_ist = day_start_ist.replace(hour=9, minute=0, second=0, microsecond=0)
                day_end_time_ist = day_start_ist.replace(hour=17, minute=0, second=0, microsecond=0)
                
                if day_start_time_ist < now_ist:
                    day_start_time_ist = now_ist

                # Sort events by start time
                events.sort(key=lambda x: dateutil.parser.parse(x['start'].get('dateTime', x['start'].get('date'))))
                
                current_free_start_ist = day_start_time_ist
                
                for event in events:
                    event_start_str = event['start'].get('dateTime', event['start'].get('date'))
                    event_end_str = event['end'].get('dateTime', event['end'].get('date'))
                    
                    event_start = dateutil.parser.parse(event_start_str)
                    event_end = dateutil.parser.parse(event_end_str)
                    
                    # Convert event times to IST
                    if event_start.tzinfo is None:
                        event_start = event_start.replace(tzinfo=timezone.utc)
                    if event_end.tzinfo is None:
                        event_end = event_end.replace(tzinfo=timezone.utc)
                    
                    event_start = event_start.astimezone(ist)
                    event_end = event_end.astimezone(ist)
                    
                    logger.info(f"Checking event: {event.get('summary')} from {event_start} to {event_end}")

                    # Check if there's enough time for a meeting before this event
                    if current_free_start_ist + timedelta(minutes=duration_minutes) <= event_start:
                        # Generate slots between current_free_start_ist and event_start
                        new_slots = self._generate_slots_in_range(
                            current_free_start_ist, 
                            event_start, 
                            duration_minutes
                        )
                        logger.info(f"Found {len(new_slots)} available slots before event")
                        available_slots.extend(new_slots)
                    
                    # Update current_free_start_ist to after this event
                    current_free_start_ist = max(current_free_start_ist, event_end)
                
                # Check for slots after the last event
                if current_free_start_ist + timedelta(minutes=duration_minutes) <= day_end_time_ist:
                    new_slots = self._generate_slots_in_range(
                        current_free_start_ist,
                        day_end_time_ist,
                        duration_minutes
                    )
                    logger.info(f"Found {len(new_slots)} available slots after last event")
                    available_slots.extend(new_slots)

            # Filter by preferred time if specified
            if preferred_time:
                filtered_slots = []
                for slot in available_slots:
                    slot_start_dt = dateutil.parser.parse(slot['start'])
                    if slot_start_dt.tzinfo is None:
                        slot_start_dt = slot_start_dt.replace(tzinfo=ist)
                    else:
                        slot_start_dt = slot_start_dt.astimezone(ist)
                    
                    slot_start_hour = slot_start_dt.hour
                    if preferred_time == 'morning' and 6 <= slot_start_hour < 12:
                        filtered_slots.append(slot)
                    elif preferred_time == 'afternoon' and 12 <= slot_start_hour < 18:
                        filtered_slots.append(slot)
                    elif preferred_time == 'evening' and 18 <= slot_start_hour <= 23:
                        filtered_slots.append(slot)
                available_slots = filtered_slots

            # Sort slots by start time
            available_slots.sort(key=lambda x: dateutil.parser.parse(x['start']))
            
            # Remove any slots that overlap with existing events
            final_slots = []
            for slot in available_slots:
                slot_start = dateutil.parser.parse(slot['start'])
                slot_end = dateutil.parser.parse(slot['end'])
                
                if slot_start.tzinfo is None:
                    slot_start = slot_start.replace(tzinfo=ist)
                if slot_end.tzinfo is None:
                    slot_end = slot_end.replace(tzinfo=ist)
                
                # Check if this slot overlaps with any existing events
                is_available = True
                for event in events:
                    event_start = dateutil.parser.parse(event['start'].get('dateTime', event['start'].get('date')))
                    event_end = dateutil.parser.parse(event['end'].get('dateTime', event['end'].get('date')))
                    
                    if event_start.tzinfo is None:
                        event_start = event_start.replace(tzinfo=timezone.utc)
                    if event_end.tzinfo is None:
                        event_end = event_end.replace(tzinfo=timezone.utc)
                    
                    event_start = event_start.astimezone(ist)
                    event_end = event_end.astimezone(ist)
                    
                    # Check for overlap
                    if (slot_start < event_end and slot_end > event_start):
                        is_available = False
                        break
                
                if is_available:
                    final_slots.append(slot)
            
            # Limit to 5 slots for brevity
            return final_slots[:5]

        except HttpError as error:
            logger.error(f"An error occurred: {error}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in find_available_slots: {str(e)}")
            return []

    def _generate_slots_in_range(self, start_time: datetime, end_time: datetime, duration_minutes: int) -> List[Dict]:
        """Helper to generate slots within a given range"""
        slots = []
        current = start_time
        
        # Round up to the next 30-minute mark
        minutes_to_add = (30 - current.minute % 30) % 30
        if minutes_to_add > 0:
            current = current + timedelta(minutes=minutes_to_add)
        
        while current + timedelta(minutes=duration_minutes) <= end_time:
            slot_end = current + timedelta(minutes=duration_minutes)
            slots.append({
                'start': current.isoformat(),
                'end': slot_end.isoformat(),
                'formatted': f"{current.strftime('%A, %B %d at %I:%M %p')} - {slot_end.strftime('%I:%M %p')} IST"
            })
            # Increment by the specified duration to avoid overlapping slots
            current += timedelta(minutes=duration_minutes)
        return slots

    def _simulate_available_slots(self, duration_minutes: int, preferred_time: str) -> List[Dict]:
        """Simulate available slots if calendar service is not available"""
        logger.info("Simulating available slots")
        simulated_slots = []
        ist = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(ist)
        
        for i in range(1, 6):  # Simulate 5 slots
            slot_start = now_ist + timedelta(days=i, hours=9, minutes=0)
            slot_end = slot_start + timedelta(minutes=duration_minutes)
            simulated_slots.append({
                'start': slot_start.isoformat(),
                'end': slot_end.isoformat(),
                'formatted': f"{slot_start.strftime('%A, %B %d at %I:%M %p')} - {slot_end.strftime('%I:%M %p')} IST"
            })
        return simulated_slots

    def create_calendar_event(self, slot: Dict, title: str = "Meeting", 
                            attendees: List[str] = None) -> bool:
        """Create a new calendar event"""
        if not self.calendar_service:
            logger.warning("Calendar service not initialized. Simulating event creation.")
            return True # Simulate success

        try:
            # Parse the slot times (which are in IST)
            ist = timezone(timedelta(hours=5, minutes=30))
            start_time = dateutil.parser.parse(slot['start'])
            end_time = dateutil.parser.parse(slot['end'])
            
            # Ensure times are in IST
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=ist)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=ist)
            
            # Convert to UTC for Google Calendar
            start_time_utc = start_time.astimezone(timezone.utc)
            end_time_utc = end_time.astimezone(timezone.utc)
            
            event = {
                'summary': title,
                'start': {'dateTime': start_time_utc.isoformat(), 'timeZone': 'UTC'},
                'end': {'dateTime': end_time_utc.isoformat(), 'timeZone': 'UTC'},
                'attendees': [{'email': email} for email in attendees] if attendees else [],
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},
                        {'method': 'popup', 'minutes': 10},
                    ],
                },
            }
            
            created_event = self.calendar_service.events().insert(
                calendarId='primary', body=event).execute()
            
            logger.info(f"Created calendar event: {created_event.get('htmlLink')}")
            return True
            
        except HttpError as error:
            logger.error(f"Failed to create calendar event: {error}")
            return False
        except Exception as e:
            logger.error(f"Error creating calendar event: {str(e)}")
            return False
    
    def generate_ai_response(self, user_message: str) -> str:
        """Generate AI response using Gemini"""
        if not self.model:
            raise ValueError("Gemini model not initialized. Please check your API key.")
        
        # Build context for the AI
        context = f"""
        You are a smart scheduling assistant. Your job is to help users schedule meetings by:
        1. Understanding their scheduling needs
        2. Asking clarifying questions when information is missing
        3. Checking calendar availability 
        4. Proposing meeting times
        5. Confirming bookings
        
        Current conversation context:
        - Duration: {self.conversation_context.get('duration', 'Not specified')}
        - Preferred time: {self.conversation_context.get('preferred_time', 'Not specified')}
        - Specific Date: {self.conversation_context.get('specific_date', 'Not specified')}
        - Title: {self.conversation_context.get('title', 'Not specified')}
        - Available slots found: {len(self.conversation_context.get('available_slots', []))}
        
        User message: "{user_message}"
        
        Based on the current context and user message, formulate a helpful and concise response. 
        If available slots are found in the context, please just acknowledge that you found them and ask the user to choose one by number. Do NOT list the slots yourself; the frontend will display them separately.
        If no slots are found, inform the user and suggest different times or parameters.
        Keep responses concise but friendly.
        """
        
        try:
            response = self.model.generate_content(context)
            return response.text.strip()
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            raise ValueError(f"Failed to generate AI response: {str(e)}")
    
    def process_message(self, user_message: str) -> Dict:
        """Process user message and generate response"""
        try:
            # Parse time information from the message
            parsed_info = self.parse_time_with_ai(user_message)
            logger.info(f"Parsed info from message: {parsed_info}")
            
            # Update conversation context with new information
            if parsed_info.get('duration'):
                self.conversation_context['duration'] = parsed_info['duration']
            if parsed_info.get('preferred_time'):
                self.conversation_context['preferred_time'] = parsed_info['preferred_time']
            if parsed_info.get('specific_date'):
                self.conversation_context['specific_date'] = parsed_info['specific_date']
            if parsed_info.get('title'):
                self.conversation_context['title'] = parsed_info['title']
            if parsed_info.get('attendees'):
                self.conversation_context['attendees'] = parsed_info['attendees']
            
            logger.info(f"Updated conversation context: {self.conversation_context}")
            
            # Determine if we need to check calendar
            should_check_calendar = (
                self.conversation_context.get('duration') and 
                (self.conversation_context.get('preferred_time') or 
                 self.conversation_context.get('specific_date'))
            )
            
            logger.info(f"Should check calendar: {should_check_calendar}")
            
            # Only check calendar if we have necessary information
            if should_check_calendar:
                duration_minutes = self._parse_duration_to_minutes(self.conversation_context['duration'])
                logger.info(f"Checking calendar for {duration_minutes} minute slots")
                
                available_slots = self.find_available_slots(
                    duration_minutes=duration_minutes,
                    preferred_time=self.conversation_context.get('preferred_time'),
                    days_ahead=7
                )
                
                self.conversation_context['available_slots'] = available_slots
                logger.info(f"Found {len(available_slots)} available slots")
                
                # Generate response based on available slots
                if available_slots:
                    ai_text_response = self.generate_ai_response(user_message)
                    response = {
                        'response': ai_text_response,
                        'context': self.conversation_context,
                        'has_slots': True,
                        'available_slots': available_slots # Ensure available_slots are sent
                    }
                else:
                    # If no slots found, ask for alternative time
                    response = {
                        'response': "I couldn't find any available slots matching your criteria. Would you like to try a different time or day?",
                        'context': self.conversation_context,
                        'has_slots': False
                    }
            else:
                # If we don't have enough information, ask for it
                missing_info = []
                if not self.conversation_context.get('duration'):
                    missing_info.append("duration")
                if not self.conversation_context.get('preferred_time') and not self.conversation_context.get('specific_date'):
                    missing_info.append("time preference")
                
                response = {
                    'response': f"To help you schedule a meeting, I need to know the {', '.join(missing_info)}. Could you please provide that information?",
                    'context': self.conversation_context,
                    'has_slots': False
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                'response': "I'm sorry, I encountered an error processing your request. Please try again.",
                'context': self.conversation_context,
                'has_slots': False
            }
    
    def _parse_duration_to_minutes(self, duration_str: str) -> int:
        """Convert duration string to minutes"""
        if not duration_str:
            return 60  # Default to 1 hour
        
        duration_str = duration_str.lower().strip()
        logger.info(f"Parsing duration string: '{duration_str}'")
        
        # Handle "half hour" or "half an hour"
        if 'half' in duration_str:
            logger.info("Detected half hour duration")
            return 30
        
        # Handle explicit minutes
        if 'minute' in duration_str or 'min' in duration_str:
            minutes = re.search(r'(\d+)', duration_str)
            if minutes:
                parsed_minutes = int(minutes.group(1))
                logger.info(f"Parsed {parsed_minutes} minutes from duration string")
                return parsed_minutes
        
        # Handle hours
        if 'hour' in duration_str:
            hours = re.search(r'(\d+)', duration_str)
            if hours:
                parsed_hours = int(hours.group(1))
                logger.info(f"Parsed {parsed_hours} hours from duration string")
                return parsed_hours * 60
        
        # Handle numeric values without units
        numbers = re.search(r'(\d+)', duration_str)
        if numbers:
            parsed_number = int(numbers.group(1))
            logger.info(f"Parsed {parsed_number} from duration string")
            return parsed_number
        
        logger.info("No duration found, defaulting to 60 minutes")
        return 60  # Default to 1 hour

# Initialize the agent
scheduler_agent = SmartSchedulerAgent()

@app.route('/')
def index():
    """Serve the main chat interface."""
    # Check if we need to authenticate
    if not scheduler_agent.calendar_service:
        auth_success = scheduler_agent.setup_calendar_service()
        if auth_success:
            # Redirect to home page after successful authentication
            return """
            <html>
                <head>
                    <title>Authentication Successful</title>
                    <script>
                        window.location.href = '/';
                    </script>
                </head>
                <body>
                    <p>Authentication successful! Redirecting to home page...</p>
                </body>
            </html>
            """
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a chat message and return the AI's response."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided', 'success': False}), 400

        message = data['message']
        logger.info(f"Received chat message: {message}")

        # Process the message and get the response
        response = scheduler_agent.process_message(message)
        logger.info(f"AI response (from process_message): {response}")

        # Check if we need to update available slots
        global last_slot_check, cached_slots
        current_time = datetime.now()
        
        if (not last_slot_check or 
            not cached_slots or 
            (current_time - last_slot_check).total_seconds() > SLOT_CACHE_DURATION):
            
            # Only check calendar if we have necessary information
            if (scheduler_agent.conversation_context.get('duration') and 
                (scheduler_agent.conversation_context.get('preferred_time') or 
                 scheduler_agent.conversation_context.get('specific_date'))):
                
                duration_minutes = scheduler_agent._parse_duration_to_minutes(
                    scheduler_agent.conversation_context['duration']
                )
                
                cached_slots = scheduler_agent.find_available_slots(
                    duration_minutes=duration_minutes,
                    preferred_time=scheduler_agent.conversation_context.get('preferred_time'),
                    days_ahead=7
                )
                last_slot_check = current_time
                
                # Update the context with cached slots
                scheduler_agent.conversation_context['available_slots'] = cached_slots
                response['context']['available_slots'] = cached_slots
                response['has_slots'] = bool(cached_slots)
                response['available_slots'] = cached_slots

        # Ensure the response is properly formatted
        if isinstance(response, dict):
            response_data = {
                'success': True,
                'response': response.get('response', ''),
                'context': response.get('context', {}),
                'has_slots': bool(response.get('available_slots')),
                'available_slots': response.get('available_slots', [])
            }
        else:
            response_data = {
                'success': True,
                'response': str(response),
                'context': scheduler_agent.conversation_context,
                'has_slots': False,
                'slots': []
            }

        logger.info(f"Sending response: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        return jsonify({
            'error': f'Failed to process message: {str(e)}',
            'success': False
        }), 500

@app.route('/api/book', methods=['POST'])
def book_meeting():
    """Book a selected meeting slot"""
    try:
        data = request.get_json()
        slot_index = data.get('slot_index', 0)
        title = data.get('title', 'Meeting')
        attendees = data.get('attendees', [])
        
        available_slots = scheduler_agent.conversation_context.get('available_slots', [])
        
        if not available_slots or slot_index >= len(available_slots):
            return jsonify({'error': 'Invalid slot selection'}), 400
        
        selected_slot = available_slots[slot_index]
        
        # Create the calendar event
        success = scheduler_agent.create_calendar_event(
            slot=selected_slot,
            title=title,
            attendees=attendees
        )
        
        if success:
            return jsonify({
                'message': f'Meeting "{title}" has been scheduled for {selected_slot["formatted"]}',
                'success': True,
                'event_details': selected_slot
            })
        else:
            return jsonify({
                'error': 'Failed to create calendar event',
                'success': False
            }), 500
            
    except Exception as e:
        logger.error(f"Book meeting error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'success': False
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Reset the conversation context"""
    scheduler_agent.conversation_context = {
        'duration': None,
        'preferred_time': None,
        'title': None,
        'attendees': [],
        'current_step': 'initial',
        'available_slots': []
    }
    
    return jsonify({
        'message': 'Conversation reset successfully',
        'success': True
    })

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    """Converts text to speech using Eleven Labs."""
    if not scheduler_agent.elevenlabs_client:
        return jsonify({'error': 'Eleven Labs not initialized'}), 500

    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        audio_stream = scheduler_agent.elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=scheduler_agent.elevenlabs_voice_id,
            voice_settings=VoiceSettings(stability=0.75, similarity_boost=0.75, style=0.0, use_speaker_boost=True),
            model_id="eleven_multilingual_v2"
        )

        # Audio is returned as an iterable of bytes, convert to single bytes object
        audio_bytes = b''
        for chunk in audio_stream:
            audio_bytes += chunk

        # Audio is returned as bytes, encode it to base64 for JSON transfer
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return jsonify({
            'audio': audio_base64,
            'success': True,
            'message': 'Text converted to speech successfully'
        })

    except Exception as e:
        logger.error(f"Eleven Labs TTS error: {e}")
        return jsonify({
            'error': f'TTS failed: {str(e)}',
            'success': False
        }), 500

@app.route('/api/speech/recognize', methods=['POST'])
def recognize_speech():
    """Speech recognition endpoint"""
    try:
        # Get audio data from request
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({'error': 'No audio file provided', 'success': False}), 400
            
        audio_file = request.files['audio']
        if not audio_file:
            logger.error("Empty audio file")
            return jsonify({'error': 'Empty audio file', 'success': False}), 400

        # Read the audio data
        audio_data = audio_file.read()
        if not audio_data:
            logger.error("No audio data read from file")
            return jsonify({'error': 'No audio data read from file', 'success': False}), 400

        logger.info(f"Received audio data of size: {len(audio_data)} bytes")

        # Convert to the format SpeechRecognition expects
        try:
            # Create AudioData object with the correct parameters
            audio = sr.AudioData(
                audio_data,
                sample_rate=16000,
                sample_width=2
            )
            logger.info("Successfully created AudioData object")
        except Exception as e:
            logger.error(f"Error converting audio data: {str(e)}")
            return jsonify({'error': f'Invalid audio format: {str(e)}', 'success': False}), 400

        # Perform speech recognition
        try:
            logger.info("Attempting speech recognition...")
            text = recognizer.recognize_google(
                audio,
                language='en-US',
                show_all=False  # Only return the most likely result
            )
            logger.info(f"Successfully recognized text: {text}")
        except sr.UnknownValueError:
            logger.warning("Speech not understood")
            return jsonify({
                'success': False,
                'error': 'Could not understand audio. Please try speaking more clearly.'
            }), 400
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Speech recognition service error: {str(e)}'
            }), 500

        # Process the recognized text
        try:
            logger.info("Processing recognized text...")
            response = scheduler_agent.process_message(text)
            logger.info("Successfully processed text")
            return jsonify({
                'success': True,
                'text': text,
                'response': response
            })
        except Exception as e:
            logger.error(f"Error processing recognized text: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Error processing recognized text: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"Speech recognition error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/calendar/today', methods=['GET'])
def get_today_meetings():
    """Get today's meetings from Google Calendar"""
    try:
        if not scheduler_agent.calendar_service:
            return jsonify({
                'success': True,
                'meetings': []
            })

        # Get today's date in IST
        ist = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(ist)
        today_start_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end_ist = today_start_ist + timedelta(days=1)

        # Convert IST times to UTC for Google Calendar API
        today_start_utc = today_start_ist.astimezone(timezone.utc)
        today_end_utc = today_end_ist.astimezone(timezone.utc)

        # Format times for Google Calendar API
        time_min = today_start_utc.isoformat()
        time_max = today_end_utc.isoformat()

        # Get events from Google Calendar
        events_result = scheduler_agent.calendar_service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        
        # Convert events to IST and format them
        meetings = []
        for event in events:
            start_time = dateutil.parser.parse(event['start'].get('dateTime', event['start'].get('date')))
            end_time = dateutil.parser.parse(event['end'].get('dateTime', event['end'].get('date')))
            
            # Ensure times are in UTC first
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            
            # Convert to IST
            start_time = start_time.astimezone(ist)
            end_time = end_time.astimezone(ist)
            
            meetings.append({
                'title': event.get('summary', 'Untitled Event'),
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'description': event.get('description', ''),
                'attendees': [attendee.get('email') for attendee in event.get('attendees', [])]
            })

        return jsonify({
            'success': True,
            'meetings': meetings
        })

    except Exception as e:
        logger.error(f"Error getting today's meetings: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Check for required environment variables
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  Warning: GEMINI_API_KEY not set. AI features will be simulated.")
    
    # Check for Google Calendar API environment variables
    if not all([os.getenv('GOOGLE_CLIENT_ID'), os.getenv('GOOGLE_CLIENT_SECRET'), os.getenv('GOOGLE_REDIRECT_URI')]):
        print("⚠️  Warning: Google Calendar API environment variables (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI) not fully set. Calendar features will be simulated.")
    
    print("🚀 Starting Smart Scheduler AI Agent...")
    print("📍 Access the app at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)