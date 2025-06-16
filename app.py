from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import asyncio
import aiohttp
import httpx
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build
import google.generativeai as genai
from elevenlabs import ElevenLabs
import json
import os
import re
import logging
from datetime import datetime, timedelta, timezone
import dateutil.parser
from typing import Dict, List, Optional
from functools import lru_cache
import warnings
from dotenv import load_dotenv
import base64
import speech_recognition as sr
from io import BytesIO
import wave
import concurrent.futures

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Constants
SCOPES = ['https://www.googleapis.com/auth/calendar.events', 'https://www.googleapis.com/auth/calendar.readonly']
SLOT_CACHE_DURATION = 300  # 5 minutes in seconds
CALENDAR_CACHE_DURATION = 3600  # 1 hour cache for calendar events
calendar_events_cache = {}
calendar_events_timestamp = None
slot_cache = {}
REDIRECT_URI = 'http://localhost:5000/oauth2callback'  # Must match exactly what's in Google Cloud Console
CLIENT_CONFIG = {
    "web": {
        "client_id": os.getenv('GOOGLE_CLIENT_ID'),
        "client_secret": os.getenv('GOOGLE_CLIENT_SECRET'),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": [REDIRECT_URI]  # Must match exactly
    }
}

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
            'available_slots': [],
            'conversation_history': []  # Add conversation history
        }
        
        # Initialize Gemini AI
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            logger.warning("GEMINI_API_KEY not found. AI features will be simulated.")
            self.model = None
        
        # Initialize Eleven Labs
        if self.eleven_labs_api_key:
            self.elevenlabs_client = ElevenLabs(api_key=self.eleven_labs_api_key)
            self.elevenlabs_voice_id = 'EXAVITQu4vr4xnSDxMaL'
            logger.info("Eleven Labs API initialized successfully")
        else:
            logger.warning("ELEVEN_LABS_API_KEY not found. Eleven Labs features will be disabled.")
            self.elevenlabs_client = None
            self.elevenlabs_voice_id = None

        # Initialize Google Calendar and caches
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(self.setup_calendar_service())
            asyncio.create_task(self._initialize_caches())
        else:
            loop.run_until_complete(self.setup_calendar_service())
            loop.run_until_complete(self._initialize_caches())

    async def setup_calendar_service(self):
        """Setup Google Calendar service asynchronously"""
        try:
            creds = None
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
                logger.info("Loaded existing credentials from token.json")
            
                if not creds or not creds.valid:
                    if creds and creds.expired and creds.refresh_token:
                        await asyncio.to_thread(creds.refresh, GoogleRequest())
                        logger.info("Refreshed expired credentials")
                    else:
                        # Redirect to auth if credentials are invalid
                        logger.error("Invalid credentials, need to re-authenticate")
                        return False
                
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
                    logger.info("Saved new credentials to token.json")
            
                self.calendar_service = build('calendar', 'v3', credentials=creds)
                
                # Verify calendar service is working
                try:
                    calendar_list = await asyncio.to_thread(self.calendar_service.calendarList().list().execute)
                    logger.info(f"Successfully connected to Google Calendar. Found {len(calendar_list.get('items', []))} calendars")
                except Exception as e:
                    logger.error(f"Error verifying calendar service: {str(e)}")
                    return False
                
                logger.info("Calendar service initialized successfully")
                return True
            
            logger.error("No token.json found, need to authenticate")
            return False
        except Exception as e:
            logger.error(f"Error setting up calendar service: {str(e)}")
            self.calendar_service = None
            return False

    async def _initialize_caches(self):
        """Initialize caches in background"""
        try:
            # Pre-fetch calendar events for a wider range
            await self._fetch_calendar_events()
            
            # Pre-calculate common slots
            common_durations = [30, 60, 90]  # 30 min, 1 hour, 1.5 hours
            common_times = ['morning', 'afternoon', 'evening']
            
            # Calculate slots for next 7 days
            for duration in common_durations:
                for time in common_times:
                    cache_key = f"{duration}_{time}"
                    if cache_key not in slot_cache:
                        await self._calculate_slots(duration, time)
                        
            logger.info(f"Cache initialized with {len(calendar_events_cache)} events")
        except Exception as e:
            logger.error(f"Error initializing caches: {e}")

    async def _fetch_calendar_events(self):
        """Fetch and cache calendar events asynchronously"""
        global calendar_events_cache, calendar_events_timestamp
        
        if not self.calendar_service:
            logger.error("Calendar service not initialized")
            return
            
        try:
            ist = timezone(timedelta(hours=5, minutes=30))
            now_ist = datetime.now(ist)
            
            # If we have a specific date in the conversation context, use that
            if self.conversation_context.get('specific_date'):
                try:
                    # Try parsing as YYYY-MM-DD first
                    target_date = datetime.strptime(self.conversation_context['specific_date'], '%Y-%m-%d')
                    logger.info(f"Specific date parsed: {target_date.date()}")
                except ValueError:
                    # If it's a day name, convert to date
                    days = {
                        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                        'friday': 4, 'saturday': 5, 'sunday': 6
                    }
                    day_name = self.conversation_context['specific_date'].lower()
                    if day_name in days:
                        target_day = days[day_name]
                        current_day = now_ist.weekday()
                        days_until = (target_day - current_day) % 7
                        if days_until == 0:  # If today is the target day
                            target_date = now_ist
                        else:
                            target_date = now_ist + timedelta(days=days_until)
                    else:
                        target_date = now_ist
                
                # Set a wider range around the target date to ensure we get all events
                start_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
                end_time = target_date.replace(hour=23, minute=59, second=59, microsecond=999999) + timedelta(days=1)
            else:
                # For general queries, fetch a wider range
                start_time = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = start_time + timedelta(days=30)  # Increased to 30 days
            
            start_time_utc = start_time.astimezone(timezone.utc)
            end_time_utc = end_time.astimezone(timezone.utc)
            
            logger.info(f"Checking calendar for slots between {start_time_utc} and {end_time_utc}")
            
            # Disable file cache warning
            warnings.filterwarnings('ignore', message='file_cache is only supported with oauth2client<4.0.0')
            
            # Create the request
            request = self.calendar_service.events().list(
                calendarId='primary',
                timeMin=start_time_utc.isoformat(),
                timeMax=end_time_utc.isoformat(),
                singleEvents=True,
                orderBy='startTime'
            )
            
            # Execute the request in a thread
            events_result = await asyncio.to_thread(request.execute)
            
            # Update cache with new events
            new_events = {
                event['id']: event for event in events_result.get('items', [])
            }
            
            # Merge with existing cache instead of replacing
            if calendar_events_cache:
                calendar_events_cache.update(new_events)
            else:
                calendar_events_cache = new_events
                
            calendar_events_timestamp = datetime.now()
            
            logger.info(f"Fetched {len(new_events)} new events, total in cache: {len(calendar_events_cache)}")
            
        except Exception as e:
            logger.error(f"Error fetching calendar events: {str(e)}")
            # Don't clear the cache on error, just log it
            if not calendar_events_cache:
                calendar_events_cache = {}

    async def _calculate_slots(self, duration_minutes: int, preferred_time: str = None) -> List[Dict]:
        """Calculate available slots using cached events asynchronously"""
        if not calendar_events_cache:
            logger.info("No calendar events in cache")
            return []
            
        ist = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(ist)
        
        # Get events for the target day
        events = list(calendar_events_cache.values())
        logger.info(f"Total events in cache: {len(events)}")
        events.sort(key=lambda x: dateutil.parser.parse(x['start'].get('dateTime', x['start'].get('date'))))
        
        available_slots = []
        
        # If we have a specific date, only look at that day
        if self.conversation_context.get('specific_date'):
            try:
                # Parse the specific date
                target_date = datetime.strptime(self.conversation_context['specific_date'], '%Y-%m-%d')
                logger.info(f"Looking for slots on: {target_date.date()}")
                
                # Make target_date timezone-aware
                target_date = target_date.replace(tzinfo=ist)
                
                # Set day boundaries in IST
                day_start_ist = target_date.replace(hour=9, minute=0, second=0, microsecond=0)
                day_end_ist = target_date.replace(hour=17, minute=0, second=0, microsecond=0)
                
                logger.info(f"Day boundaries - Start: {day_start_ist}, End: {day_end_ist}")
                
                # Convert to UTC for Google Calendar API
                day_start_utc = day_start_ist.astimezone(timezone.utc)
                day_end_utc = day_end_ist.astimezone(timezone.utc)
                
                # Filter events for this day
                day_events = [
                    e for e in events 
                    if dateutil.parser.parse(e['start'].get('dateTime', e['start'].get('date'))).date() == target_date.date()
                ]
                
                logger.info(f"Found {len(day_events)} events for this day")
                
                current_free_start = day_start_ist
                for event in day_events:
                    event_start = dateutil.parser.parse(event['start'].get('dateTime', event['start'].get('date')))
                    event_end = dateutil.parser.parse(event['end'].get('dateTime', event['end'].get('date')))
                    
                    # Ensure event times are timezone-aware
                    if event_start.tzinfo is None:
                        event_start = event_start.replace(tzinfo=timezone.utc)
                    if event_end.tzinfo is None:
                        event_end = event_end.replace(tzinfo=timezone.utc)
                    
                    # Convert to IST for comparison
                    event_start = event_start.astimezone(ist)
                    event_end = event_end.astimezone(ist)
                    
                    logger.info(f"Processing event: {event_start} - {event_end}")
                    
                    if current_free_start + timedelta(minutes=duration_minutes) <= event_start:
                        new_slots = await self._generate_slots_in_range(
                            current_free_start,
                            event_start,
                            duration_minutes
                        )
                        logger.info(f"Found {len(new_slots)} slots before event")
                        available_slots.extend(new_slots)
                    
                    current_free_start = max(current_free_start, event_end)
                
                if current_free_start + timedelta(minutes=duration_minutes) <= day_end_ist:
                    new_slots = await self._generate_slots_in_range(
                        current_free_start,
                        day_end_ist,
                        duration_minutes
                    )
                    logger.info(f"Found {len(new_slots)} slots after last event")
                    available_slots.extend(new_slots)
                
                logger.info(f"Total available slots found: {len(available_slots)}")
                
            except ValueError as e:
                logger.error(f"Invalid date format: {self.conversation_context['specific_date']} - {str(e)}")
                return []
        
        # Filter by preferred time
        if preferred_time:
            filtered_slots = []
            for slot in available_slots:
                slot_start = dateutil.parser.parse(slot['start'])
                # Ensure slot time is timezone-aware
                if slot_start.tzinfo is None:
                    slot_start = slot_start.replace(tzinfo=ist)
                else:
                    slot_start = slot_start.astimezone(ist)
                
                slot_hour = slot_start.hour
                logger.info(f"Checking slot hour: {slot_hour} for preferred time: {preferred_time}")
                
                if preferred_time == 'morning' and 6 <= slot_hour < 12:
                    filtered_slots.append(slot)
                elif preferred_time == 'afternoon' and 12 <= slot_hour < 18:
                    filtered_slots.append(slot)
                elif preferred_time == 'evening' and 18 <= slot_hour <= 23:
                    filtered_slots.append(slot)
            
            logger.info(f"Filtered slots for {preferred_time}: {len(filtered_slots)} slots")
            available_slots = filtered_slots
        
        return available_slots[:5]  # Limit to 5 slots

    async def _generate_slots_in_range(self, start_time: datetime, end_time: datetime, duration_minutes: int) -> List[Dict]:
        """Generate available slots in a time range asynchronously"""
        ist = timezone(timedelta(hours=5, minutes=30))
        
        # Ensure both times are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=ist)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=ist)
        
        # Convert to IST for consistency
        start_time = start_time.astimezone(ist)
        end_time = end_time.astimezone(ist)
        
        slots = []
        current_time = start_time
        
        while current_time + timedelta(minutes=duration_minutes) <= end_time:
            slot_end = current_time + timedelta(minutes=duration_minutes)
            slots.append({
                'start': current_time.isoformat(),
                'end': slot_end.isoformat(),
                'formatted': f"{current_time.strftime('%A, %B %d at %I:%M %p')} - {slot_end.strftime('%I:%M %p')} IST"
            })
            current_time += timedelta(minutes=30)  # 30-minute intervals
        
        return slots

    async def find_available_slots(self, duration_minutes: int = 60, 
                                 preferred_time: str = None, 
                                 days_ahead: int = 7) -> List[Dict]:
        """Find available time slots using cached data asynchronously"""
        logger.info(f"Finding available slots for {duration_minutes} minutes")
        
        # Check cache first
        cache_key = f"{duration_minutes}_{preferred_time}"
        if cache_key in slot_cache:
            cached_result = slot_cache[cache_key]
            if (datetime.now() - cached_result['timestamp']).total_seconds() < SLOT_CACHE_DURATION:
                logger.info("Returning cached slots")
                return cached_result['slots']
        
        # Check if we need to refresh calendar events
        global calendar_events_timestamp
        if (not calendar_events_timestamp or 
            (datetime.now() - calendar_events_timestamp).total_seconds() > CALENDAR_CACHE_DURATION):
            await self._fetch_calendar_events()
        
        # Calculate slots using cached events
        available_slots = await self._calculate_slots(duration_minutes, preferred_time)
        
        # Cache the results
        slot_cache[cache_key] = {
            'slots': available_slots,
            'timestamp': datetime.now()
        }
        
        return available_slots

    async def process_message(self, user_message: str) -> Dict:
        """Process user message and generate response asynchronously"""
        try:
            # Add user message to conversation history
            self.conversation_context['conversation_history'].append({
                'role': 'user',
                'content': user_message
            })

            # Parse time information from the message
            parsed_info = await self.parse_time_with_ai(user_message)
            logger.info(f"Parsed info from message: {parsed_info}")
            
            if not parsed_info:
                response_text = await self._generate_ai_response(
                    "I couldn't understand the time information. Could you please rephrase your request?",
                    user_message
                )
                return {
                    'response': response_text,
                    'context': self.conversation_context,
                    'has_slots': False
                }
            
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
            
            # Determine if we need to check calendar
            should_check_calendar = (
                self.conversation_context.get('duration') and 
                (self.conversation_context.get('preferred_time') or 
                 self.conversation_context.get('specific_date'))
            )
            
            # Only check calendar if we have necessary information
            if should_check_calendar:
                try:
                    duration_minutes = self._parse_duration_to_minutes(self.conversation_context['duration'])
                    
                    # Always fetch fresh calendar data
                    await self._fetch_calendar_events()
                    
                    available_slots = await self._calculate_slots(
                        duration_minutes=duration_minutes,
                        preferred_time=self.conversation_context.get('preferred_time')
                    )
                    
                    self.conversation_context['available_slots'] = available_slots
                    
                    # Generate response based on available slots
                    if available_slots:
                        response_text = await self._generate_ai_response(
                            "I found some available slots. Please choose your preferred time from the options displayed.",
                            user_message,
                            available_slots=available_slots
                        )
                        response = {
                            'response': response_text,
                            'context': self.conversation_context,
                            'has_slots': True,
                            'available_slots': available_slots
                        }
                    else:
                        response_text = await self._generate_ai_response(
                            "I couldn't find any available slots matching your criteria. Would you like to try a different time or day?",
                            user_message
                        )
                        response = {
                            'response': response_text,
                            'context': self.conversation_context,
                            'has_slots': False
                        }
                except Exception as e:
                    logger.error(f"Error checking calendar: {str(e)}", exc_info=True)
                    response_text = await self._generate_ai_response(
                        "I encountered an error while checking the calendar. Please try again.",
                        user_message
                    )
                    return {
                        'response': response_text,
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
                
                response_text = await self._generate_ai_response(
                    f"To help you schedule a meeting, I need to know the {', '.join(missing_info)}. Could you please provide that information?",
                    user_message
                )
                response = {
                    'response': response_text,
                    'context': self.conversation_context,
                    'has_slots': False
                }
            
            # Add AI response to conversation history
            self.conversation_context['conversation_history'].append({
                'role': 'assistant',
                'content': response['response']
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            response_text = await self._generate_ai_response(
                f"I'm sorry, I encountered an error: {str(e)}",
                user_message
            )
            return {
                'response': response_text,
                'context': self.conversation_context,
                'has_slots': False
            }

    def _parse_duration_to_minutes(self, duration_str: str) -> int:
        """Convert duration string to minutes"""
        try:
            if not duration_str:
                return 60  # Default to 1 hour
            
            # Extract numbers and units
            match = re.search(r'(\d+)\s*(hour|hr|minute|min)s?', duration_str.lower())
            if not match:
                return 60  # Default to 1 hour if no match
                
            number = int(match.group(1))
            unit = match.group(2)
            
            if unit.startswith('hour') or unit == 'hr':
                return number * 60
            else:
                return number
                
        except Exception as e:
            logger.error(f"Error parsing duration: {str(e)}")
            return 60  # Default to 1 hour on error

    async def parse_time_with_ai(self, user_input: str) -> Dict:
        """Use AI to parse complex time expressions with caching"""
        if not self.model:
            logger.info("Using fallback time parsing (no AI model available)")
            return self._fallback_time_parsing(user_input)
        
        # Get current date for reference
        current_date = datetime.now()
        
        # First try quick regex patterns for common cases
        quick_patterns = {
            r'morning': {'preferred_time': 'morning'},
            r'afternoon': {'preferred_time': 'afternoon'},
            r'evening': {'preferred_time': 'evening'},
            r'noon': {'preferred_time': 'afternoon'},
            r'(\d+)\s*hours?': lambda m: {'duration': f"{m.group(1)} hour{'s' if int(m.group(1)) > 1 else ''}"},
            r'(\d+)\s*hrs?': lambda m: {'duration': f"{m.group(1)} hour{'s' if int(m.group(1)) > 1 else ''}"},
            r'(\d+)\s*minutes?': lambda m: {'duration': f"{m.group(1)} minutes"},
            r'(\d+)\s*mins?': lambda m: {'duration': f"{m.group(1)} minutes"},
            r'tomorrow': {'specific_date': (current_date + timedelta(days=1)).strftime('%Y-%m-%d')},
            r'next week': {'specific_date': (current_date + timedelta(days=7)).strftime('%Y-%m-%d')},
            r'next (\w+)': lambda m: self._parse_next_day(m.group(1)),
            r'this (\w+)': lambda m: self._parse_this_day(m.group(1)),
            r'on (\w+)': lambda m: self._parse_next_day(m.group(1)),
            r'asap': {'urgency': 'asap'},
            r'urgent': {'urgency': 'asap'},
            r'immediately': {'urgency': 'asap'},
            r'flexible': {'urgency': 'flexible'}
        }
        
        text = user_input.lower()
        result = {}
        
        # Try all quick patterns and combine results
        for pattern, value in quick_patterns.items():
            match = re.search(pattern, text)
            if match:
                if callable(value):
                    result.update(value(match))
                else:
                    result.update(value)
        
        # Enhanced AI prompt for better date parsing
        prompt = f"""
        Parse this scheduling request and extract the key information:
        "{user_input}"
        
        Current date is: {current_date.strftime('%Y-%m-%d')}
        
        Return a JSON object with these fields:
        - duration: meeting duration (e.g., "30 minutes", "1 hour", "2 hours")
        - preferred_time: general time preference (e.g., "morning", "afternoon", "evening")
        - specific_date: specific date in YYYY-MM-DD format. For relative dates:
          * "tomorrow" -> {current_date + timedelta(days=1)}
          * "next [day]" -> next occurrence of that day from {current_date}
          * "this [day]" -> this week's occurrence from {current_date}
          * "next week" -> {current_date + timedelta(days=7)}
        - urgency: how urgent (e.g., "asap", "flexible", "specific")
        - meeting_type: type of meeting if mentioned
        
        Important rules:
        1. Always use dates relative to {current_date.strftime('%Y-%m-%d')}
        2. For "next [day]", calculate the next occurrence of that day from {current_date}
        3. For "this [day]", calculate this week's occurrence from {current_date}
        4. If no specific date is mentioned, don't include specific_date field
        5. Only include fields that are clearly mentioned
        6. Return valid JSON only
        
        Example responses (using current date {current_date.strftime('%Y-%m-%d')}):
        - "Schedule a meeting tomorrow afternoon": {{"specific_date": "{(current_date + timedelta(days=1)).strftime('%Y-%m-%d')}", "preferred_time": "afternoon"}}
        - "1 hour meeting next friday": {{"duration": "1 hour", "specific_date": "{self._get_next_weekday(current_date, 4).strftime('%Y-%m-%d')}"}}
        - "30 min meeting this wednesday": {{"duration": "30 minutes", "specific_date": "{self._get_this_weekday(current_date, 2).strftime('%Y-%m-%d')}"}}
        """
        
        try:
            logger.info("Sending request to Gemini AI")
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            logger.info(f"Raw AI response: {response.text}")
            
            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            
            if not response_text.startswith('{'):
                response_text = '{' + response_text
            if not response_text.endswith('}'):
                response_text = response_text + '}'
            
            response_text = re.sub(r',\s*}', '}', response_text)
            
            parsed_data = json.loads(response_text)
            logger.info(f"Parsed AI response: {parsed_data}")
            
            # Convert day names to actual dates if needed
            if parsed_data.get('specific_date'):
                specific_date = parsed_data['specific_date']
                if specific_date.lower() in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                    parsed_data['specific_date'] = self._parse_next_day(specific_date.lower())['specific_date']
                elif specific_date.startswith('next '):
                    day_name = specific_date[5:].lower()
                    parsed_data['specific_date'] = self._parse_next_day(day_name)['specific_date']
                elif specific_date.startswith('this '):
                    day_name = specific_date[5:].lower()
                    parsed_data['specific_date'] = self._parse_this_day(day_name)['specific_date']
            
            # Combine AI results with quick pattern results
            result.update(parsed_data)
            return result
        except Exception as e:
            logger.error(f"AI parsing failed: {str(e)}")
            logger.info("Falling back to basic time parsing")
            return self._fallback_time_parsing(user_input)

    def _get_next_weekday(self, current_date: datetime, target_weekday: int) -> datetime:
        """Get the next occurrence of a weekday from the current date"""
        days_ahead = target_weekday - current_date.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return current_date + timedelta(days=days_ahead)

    def _get_this_weekday(self, current_date: datetime, target_weekday: int) -> datetime:
        """Get this week's occurrence of a weekday from the current date"""
        days_ahead = target_weekday - current_date.weekday()
        if days_ahead < 0:
            days_ahead += 7
        return current_date + timedelta(days=days_ahead)

    def _parse_next_day(self, day_name: str) -> Dict:
        """Parse 'next [day]' expressions"""
        days = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        day_name = day_name.lower()
        if day_name not in days:
            return {}
            
        current_date = datetime.now()
        target_day = days[day_name]
        
        # Get next occurrence of the day
        next_date = self._get_next_weekday(current_date, target_day)
        return {'specific_date': next_date.strftime('%Y-%m-%d')}

    def _parse_this_day(self, day_name: str) -> Dict:
        """Parse 'this [day]' expressions"""
        days = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        day_name = day_name.lower()
        if day_name not in days:
            return {}
            
        current_date = datetime.now()
        target_day = days[day_name]
        
        # Get this week's occurrence of the day
        this_date = self._get_this_weekday(current_date, target_day)
        return {'specific_date': this_date.strftime('%Y-%m-%d')}

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
        
        # Parse day names
        days = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        for day_name in days.keys():
            if day_name in text:
                today = datetime.now()
                target_day = days[day_name]
                current_day = today.weekday()
                days_until = (target_day - current_day) % 7
                if days_until == 0:  # If today is the target day
                    target_date = today
                else:
                    target_date = today + timedelta(days=days_until)
                result['specific_date'] = target_date.strftime('%Y-%m-%d')
                logger.info(f"Found specific date: {result['specific_date']}")
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

    async def _generate_ai_response(self, base_response: str, user_message: str, available_slots: List[Dict] = None) -> str:
        """Generate a more natural AI response using conversation history"""
        if not self.model:
            return base_response

        # Prepare conversation history for context
        conversation_context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in self.conversation_context['conversation_history'][-5:]  # Last 5 messages for context
        ])

        # Prepare slot information if available
        slot_info = ""
        if available_slots:
            slot_info = "\nAvailable time slots:\n" + "\n".join([
                f"- {slot['formatted']}"
                for slot in available_slots[:5]  # Show first 5 slots
            ])

        prompt = f"""
        You are a friendly and helpful AI scheduling assistant. You're having a conversation with a user about scheduling a meeting.
        
        Previous conversation:
        {conversation_context}
        
        Current context:
        - Duration: {self.conversation_context.get('duration', 'Not specified')}
        - Preferred time: {self.conversation_context.get('preferred_time', 'Not specified')}
        - Specific date: {self.conversation_context.get('specific_date', 'Not specified')}
        - Meeting title: {self.conversation_context.get('title', 'Not specified')}
        {slot_info}
        
        Base response: {base_response}
        
        Generate a natural, conversational response that:
        1. Maintains context from the previous conversation
        2. Sounds friendly and helpful
        3. Includes relevant information from the current context
        4. Keeps responses concise and natural
        5. Avoids technical details and URLs
        6. Uses natural language for time slots (e.g., "Here are some times that work" instead of listing raw data)
        7. Focuses on the essential information the user needs to know
        
        Make the response sound like a natural conversation, not a technical report.
        """

        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return base_response

# Initialize the agent
scheduler_agent = SmartSchedulerAgent()

@app.get("/")
async def index(request: Request):
    """Root endpoint that checks authentication and redirects if needed"""
    if not scheduler_agent.calendar_service:
        return RedirectResponse(url="/auth")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        message = data.get('message', '')
        logger.info(f"Received chat message: {message}")
        
        if not message:
            return JSONResponse({
                'success': False,
                'error': "No message provided"
            })
        
        response = await scheduler_agent.process_message(message)
        logger.info(f"AI response (from process_message): {response}")
        
        if not response:
            return JSONResponse({
                'success': False,
                'error': "No response generated"
            })
        
        return JSONResponse({
            'success': True,
            'response': response.get('response', "I'm sorry, I couldn't process that request."),
            'context': response.get('context', {}),
            'has_slots': response.get('has_slots', False),
            'available_slots': response.get('available_slots', [])
        })
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return JSONResponse({
            'success': False,
            'error': f"I'm sorry, I encountered an error: {str(e)}"
        })

@app.post("/api/book")
async def book_meeting(request: Request):
    try:
        data = await request.json()
        slot_index = data.get('slot_index')
        title = data.get('title', 'Meeting')
        attendees = data.get('attendees', [])
        
        if slot_index is None or not scheduler_agent.conversation_context.get('available_slots'):
            return JSONResponse({
                'success': False,
                'error': "Invalid slot selection"
            })
        
        slot = scheduler_agent.conversation_context['available_slots'][slot_index]
        
        # Create calendar event
        event = {
            'summary': title,
            'start': {
                'dateTime': slot['start'],
                'timeZone': 'Asia/Kolkata',
            },
            'end': {
                'dateTime': slot['end'],
                'timeZone': 'Asia/Kolkata',
            },
            'attendees': [{'email': email} for email in attendees],
        }
        
        # Create the request
        request = scheduler_agent.calendar_service.events().insert(
            calendarId='primary',
            body=event
        )
        
        # Execute the request in a thread
        event = await asyncio.to_thread(request.execute)
        
        logger.info(f"Created calendar event: {event.get('htmlLink')}")
        
        return JSONResponse({
            'success': True,
            'message': f"Meeting scheduled successfully! You can view it here: {event.get('htmlLink')}"
        })
    except Exception as e:
        logger.error(f"Error in book_meeting endpoint: {str(e)}")
        return JSONResponse({
            'success': False,
            'error': "I'm sorry, I couldn't book that slot. Please try again."
        })

@app.post("/api/reset")
async def reset_conversation():
    try:
        scheduler_agent.conversation_context = {
            'duration': None,
            'preferred_time': None,
            'title': None,
            'attendees': [],
            'current_step': 'initial',
            'available_slots': [],
            'conversation_history': []  # Reset conversation history
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
    """Convert text to speech using Eleven Labs API with streaming"""
    try:
        data = await request.json()
        text = data.get('text', '').strip()
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No text provided"}
            )
            
        if not scheduler_agent.elevenlabs_client:
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": "TTS service unavailable"}
            )
            
        # Clean up text for speech
        text = clean_text_for_speech(text)
        
        # Prepare the request to Eleven Labs API
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{scheduler_agent.elevenlabs_voice_id}"
        headers = {
            "xi-api-key": scheduler_agent.eleven_labs_api_key,
            "accept": "audio/mpeg",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "optimize_streaming_latency": 4,
            "output_format": "mp3_44100_128",
            "style": 0,
            "use_speaker_boost": False,
            "stability": 0.5,
            "similarity_boost": 0.5
        }
        
        # Use httpx for streaming
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            
            if response.status_code != 200:
                error_data = response.json()
                raise Exception(f"Eleven Labs API error: {error_data.get('detail', 'Unknown error')}")
            
            # Return the audio data directly
            return Response(
                content=response.content,
                media_type="audio/mpeg",
                headers={
                    "Content-Type": "audio/mpeg",
                    "Content-Length": str(len(response.content))
                }
            )
                
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

def clean_text_for_speech(text: str) -> str:
    """Clean text to make it more natural for speech"""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove redundant information
    text = re.sub(r'You can view it here:', '', text)
    text = re.sub(r'Meeting scheduled successfully!', 'Meeting scheduled!', text)
    
    # Clean up time slots presentation
    if "Available time slots:" in text:
        text = re.sub(r'Available time slots:.*?(?=\n\n|$)', 
                     'Here are some available times:', text, flags=re.DOTALL)
    
    # Remove redundant formatting
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Make it more conversational
    text = text.replace('IST', '')
    text = text.replace('UTC', '')
    
    # Clean up any remaining technical details
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    return text.strip()

@app.get("/api/calendar/today")
async def get_today_meetings():
    try:
        if not scheduler_agent.calendar_service:
            return JSONResponse({
                'success': False,
                'error': "Calendar service not available"
            })
        
        ist = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(ist)
        start_time = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        start_time_utc = start_time.astimezone(timezone.utc)
        end_time_utc = end_time.astimezone(timezone.utc)
        
        # Create the request
        request = scheduler_agent.calendar_service.events().list(
            calendarId='primary',
            timeMin=start_time_utc.isoformat(),
            timeMax=end_time_utc.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        )
        
        # Execute the request in a thread
        events_result = await asyncio.to_thread(request.execute)

        meetings = []
        for event in events_result.get('items', []):
            start = dateutil.parser.parse(event['start'].get('dateTime', event['start'].get('date')))
            end = dateutil.parser.parse(event['end'].get('dateTime', event['end'].get('date')))
            
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            
            start = start.astimezone(ist)
            end = end.astimezone(ist)
            
            meetings.append({
                'title': event.get('summary', 'Untitled Event'),
                'start': start.isoformat(),
                'end': end.isoformat()
            })
        
        return JSONResponse({
            'success': True,
            'meetings': meetings
        })
    except Exception as e:
        logger.error(f"Error in get_today_meetings endpoint: {str(e)}")
        return JSONResponse({
            'success': False,
            'error': "Error fetching today's meetings"
        })

@app.post("/api/speech/recognize")
async def recognize_speech(request: Request):
    try:
        form = await request.form()
        audio_file = form.get('audio')
        
        if not audio_file:
            return JSONResponse({
                'success': False,
                'error': "No audio file provided"
            })
        
        # Read the audio file
        audio_data = await audio_file.read()
        
        # Create a BytesIO object from the audio data
        audio_io = BytesIO(audio_data)
        
        # Initialize recognizer with optimized settings
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300  # Lower threshold for better recognition
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.5  # Shorter pause threshold
        recognizer.phrase_threshold = 0.3  # Lower phrase threshold for faster response
        recognizer.non_speaking_duration = 0.3  # Shorter non-speaking duration
        
        # Read the WAV file
        with wave.open(audio_io, 'rb') as wav_file:
            # Get audio data
            audio_data = wav_file.readframes(wav_file.getnframes())
            
            # Create AudioData object with optimized settings
            audio = sr.AudioData(
                audio_data,
                sample_rate=wav_file.getframerate(),
                sample_width=wav_file.getsampwidth()
            )
            
            # Process recognition and chat in parallel
            async def process_recognition():
                try:
                    # Use a thread pool for parallel processing
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # Start speech recognition in a separate thread
                        future = executor.submit(
                            recognizer.recognize_google,
                            audio,
                            language='en-US',
                            show_all=False
                        )
                        return await asyncio.wrap_future(future)
                except sr.UnknownValueError:
                    raise HTTPException(status_code=400, detail="Could not understand audio")
                except sr.RequestError as e:
                    raise HTTPException(status_code=500, detail=f"Could not request results; {str(e)}")
            
            # Get recognition result
            text = await process_recognition()
            
            # Process the recognized text through the chat endpoint
            response = await scheduler_agent.process_message(text)
            
            return JSONResponse({
                'success': True,
                'text': text,
                'response': response
            })
            
    except HTTPException as e:
        return JSONResponse({
            'success': False,
            'error': str(e.detail)
        })
    except Exception as e:
        logger.error(f"Error in speech recognition: {str(e)}")
        return JSONResponse({
            'success': False,
            'error': "Error processing speech"
        })

@app.get("/auth")
async def auth():
    """Start the OAuth2 flow"""
    try:
        # Only remove token if we're starting a new auth flow
        if os.path.exists('token.json'):
            os.remove('token.json')
            logger.info("Removed existing token.json")
        
        flow = Flow.from_client_config(
            CLIENT_CONFIG,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent',
            force_approval_prompt='force'
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
    """Handle the OAuth2 callback"""
    try:
        # Get the authorization code from the request
        code = request.query_params.get('code')
        if not code:
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
        with open('token.json', 'w') as token:
            token.write(credentials.to_json())
        
        # Initialize the calendar service
        scheduler_agent.calendar_service = build('calendar', 'v3', credentials=credentials)
        
        # Initialize caches after successful authentication
        await scheduler_agent._initialize_caches()
        
        # Redirect to the main page
        return RedirectResponse(url="/")
    except Exception as e:
        logger.error(f"Error in oauth2callback: {str(e)}")
        # If there's an error, redirect back to auth to start fresh
        return RedirectResponse(url="/auth")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)