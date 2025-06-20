import os.path
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import pytz
from dateutil import parser
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field
import config
import re

class CalendarToolInput(BaseModel):
    action: str = Field(description="Action to perform: 'check_availability', 'create_meeting', 'find_next_available', or 'get_events'")
    start_time: Optional[str] = Field(None, description="Start time in ISO format or natural language")
    end_time: Optional[str] = Field(None, description="End time in ISO format or natural language")
    duration_minutes: Optional[int] = Field(None, description="Duration in minutes")
    title: Optional[str] = Field(None, description="Meeting title")
    description: Optional[str] = Field(None, description="Meeting description")
    date_range_start: Optional[str] = Field(None, description="Start of date range to search")
    date_range_end: Optional[str] = Field(None, description="End of date range to search")
    query: Optional[str] = Field(None, description="Query string for searching events (e.g., meeting title, participant name)")

class CalendarTool:
    """A tool for interacting with Google Calendar"""
    
    def __init__(self):
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Calendar API"""
        creds = None
        token_path = 'token.json'
        
        try:
            # Load existing token
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, config.SCOPES)
            
            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except Exception as e:
                        print(f"Error refreshing credentials: {e}")
                        # If refresh fails, we'll need to re-authenticate
                        creds = None
                
                if not creds:
                    # If credentials are still not available, it means token.json is missing or invalid.
                    # At this point, we expect main.py to handle the full OAuth flow via web redirect.
                    print("Authentication token (token.json) not found or invalid. Calendar service will be unavailable until authenticated via web interface.")
                    self.service = None # Indicate service is not ready
                    return # Exit without raising an error
                
            # Save the credentials only if they were newly obtained or refreshed
            # The primary saving is in main.py, but this ensures refresh tokens are saved if refreshed by CalendarTool itself
            if creds and (not os.path.exists(token_path) or os.path.getsize(token_path) == 0 or not Credentials.from_authorized_user_file(token_path, config.SCOPES).valid):
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
                    print("Credentials saved/updated to token.json by calendar_tool (after refresh or initial load).")

            self.service = build('calendar', 'v3', credentials=creds)
            
        except Exception as e:
            print(f"Error during Google Calendar authentication: {e}")
            # Initialize service as None to indicate authentication failure
            self.service = None
            # Do NOT raise here, as we want to allow agent to continue even without calendar
            # The agent will check self.calendar_tool.service before invoking.
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime object"""
        if not time_str:
            return None
        
        try:
            # Set IST timezone
            ist = pytz.timezone('Asia/Kolkata')
            
            # Try to parse as ISO format first
            if 'T' in time_str:
                parsed_time = parser.parse(time_str)
                if parsed_time.tzinfo is None:
                    parsed_time = ist.localize(parsed_time)
                return parsed_time
            else:
                # Handle natural language time parsing
                now = datetime.now(ist)
                
                if 'tomorrow' in time_str.lower():
                    base_date = now + timedelta(days=1)
                elif 'today' in time_str.lower():
                    base_date = now
                elif 'next week' in time_str.lower():
                    base_date = now + timedelta(days=7)
                else:
                    # Try to parse with dateutil
                    parsed_time = parser.parse(time_str)
                    if parsed_time.tzinfo is None:
                        parsed_time = ist.localize(parsed_time)
                    return parsed_time
                
                # Extract time if specified
                if 'morning' in time_str.lower():
                    return base_date.replace(hour=9, minute=0, second=0, microsecond=0)
                elif 'afternoon' in time_str.lower():
                    return base_date.replace(hour=14, minute=0, second=0, microsecond=0)
                elif 'evening' in time_str.lower():
                    return base_date.replace(hour=18, minute=0, second=0, microsecond=0)
                else:
                    return base_date.replace(hour=9, minute=0, second=0, microsecond=0)
        
        except Exception as e:
            # Fallback to current time in IST
            ist = pytz.timezone('Asia/Kolkata')
            return datetime.now(ist)
    
    def _check_availability(self, start_time: datetime, end_time: datetime) -> Dict:
        """Check if a time slot is available"""
        try:
            # Query for events in the specified time range
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=start_time.isoformat() + 'Z',
                timeMax=end_time.isoformat() + 'Z',
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            if not events:
                return {
                    "available": True,
                    "message": f"Time slot from {start_time.strftime('%Y-%m-%d %I:%M %p')} to {end_time.strftime('%I:%M %p')} is available."
                }
            else:
                return {
                    "available": False,
                    "message": f"Time slot is busy. Found {len(events)} conflicting event(s).",
                    "conflicts": [event.get('summary', 'No title') for event in events]
                }
        
        except HttpError as error:
            return {"error": f"An error occurred: {error}"}
    
    def _create_meeting(self, start_time: datetime, end_time: datetime, title: str, description: str = "") -> Dict:
        """Create a new meeting"""
        try:
            # Convert to IST timezone
            ist = pytz.timezone('Asia/Kolkata')
            if start_time.tzinfo is None:
                start_time = ist.localize(start_time)
            else:
                start_time = start_time.astimezone(ist)
            
            if end_time.tzinfo is None:
                end_time = ist.localize(end_time)
            else:
                end_time = end_time.astimezone(ist)
            
            event = {
                'summary': title,
                'description': description,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'Asia/Kolkata',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'Asia/Kolkata',
                },
            }
            
            event = self.service.events().insert(calendarId='primary', body=event).execute()
            return {
                "success": True,
                "message": f"Meeting '{title}' created successfully.",
                "event_id": event.get('id'),
                "event_link": event.get('htmlLink')
            }
        
        except HttpError as error:
            return {"error": f"Failed to create meeting: {error}"}
    
    def _get_events(self, query: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict:
        """Get existing calendar events based on a query and optional time range"""
        try:
            ist = pytz.timezone('Asia/Kolkata')
            now_ist = datetime.now(ist)

            # If start_time and end_time are not provided, default to today + 14 days (full day coverage)
            if start_time is None:
                start_time = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
            if end_time is None:
                end_time = now_ist + timedelta(days=14)

            # Ensure times are timezone-aware in IST first if they aren't already
            if start_time.tzinfo is None: start_time = ist.localize(start_time)
            if end_time.tzinfo is None: end_time = ist.localize(end_time)

            # Convert to UTC for the Google Calendar API call
            start_time_utc = start_time.astimezone(pytz.utc)
            end_time_utc = end_time.astimezone(pytz.utc)
            
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=start_time_utc.isoformat(), # Use UTC time for API
                timeMax=end_time_utc.isoformat(),   # Use UTC time for API
                singleEvents=True,
                orderBy='startTime',
                maxResults=250 # Increased limit to ensure all events are fetched for client-side filtering
            ).execute()
            
            # --- NEW DEBUG LOGGING ---
            print(f"[DEBUG] Google Calendar API raw events_result for query '{query}': {json.dumps(events_result, indent=2)}")
            # --- END NEW DEBUG LOGGING ---

            events = events_result.get('items', [])
            
            # Client-side filtering if a query was provided
            if query:
                filtered_events = []
                for event in events:
                    summary = event.get('summary', '').lower()
                    description = event.get('description', '').lower()
                    # Simple keyword matching (can be expanded to regex if needed)
                    if query.lower() in summary or query.lower() in description:
                        filtered_events.append(event)
                events = filtered_events
                print(f"[DEBUG] Client-side filtered events for query '{query}': {len(events)} events found.")

            if not events:
                return {"success": True, "message": f"No events found matching '{query}'.", "events": []}
            
            formatted_events = []
            for event in events:
                event_start_raw = event['start'].get('dateTime', event['start'].get('date'))
                event_end_raw = event['end'].get('dateTime', event['end'].get('date'))
                
                event_start = parser.parse(event_start_raw)
                event_end = parser.parse(event_end_raw)
                
                if event_start.tzinfo is None:
                    event_start = ist.localize(event_start)
                else:
                    event_start = event_start.astimezone(ist)
                
                if event_end.tzinfo is None:
                    event_end = ist.localize(event_end)
                else:
                    event_end = event_end.astimezone(ist)

                formatted_events.append({
                    "summary": event.get('summary', 'No Title'),
                    "start": event_start.isoformat(),
                    "end": event_end.isoformat(),
                    "link": event.get('htmlLink')
                })
            
            return {
                "success": True,
                "message": f"Found {len(formatted_events)} event(s) matching '{query}'.",
                "events": formatted_events
            }
        
        except HttpError as error:
            return {"error": f"An error occurred while searching for events: {error}"}

    def _get_next_weekday(self, current_date: datetime, target_weekday: int, include_today: bool = True) -> datetime:
        days_ahead = target_weekday - current_date.weekday()
        if days_ahead < 0 or (days_ahead == 0 and not include_today):
            days_ahead += 7
        return current_date + timedelta(days=days_ahead)

    def _get_last_workday_of_month(self, current_date: datetime) -> datetime:
        """Calculate the last workday of the month for a given date."""
        # Get the last day of the month
        # Add a buffer of 4 days to ensure we are in the next month, then subtract its day to get last day of current month
        next_month = current_date.replace(day=28) + timedelta(days=4)
        last_day = next_month - timedelta(days=next_month.day)

        # Check if the last day is a weekend and adjust backward
        if last_day.weekday() == 5:  # Saturday (5)
            last_day -= timedelta(days=1)  # Move to Friday
        elif last_day.weekday() == 6:  # Sunday (6)
            last_day -= timedelta(days=2)  # Move to Friday

        return last_day

    def _parse_date_with_ai(self, date_str: str, current_date: datetime) -> Optional[Dict[str, Any]]:
        """Use the LLM to parse a natural language date string into a structured intent dictionary.
        The current_date parameter provides the context for relative date parsing.
        """
        try:
            import google.generativeai as genai
            ist = pytz.timezone('Asia/Kolkata')
            # Use the provided current_date for context
            now = current_date
            
            # CRITICAL: Reiterate current date forcefully to prevent LLM year hallucination
            prompt = f"""
            Today is {now.strftime('%A, %B %d, %Y, %I:%M %p')} (IST, UTC+05:30).

            Parse the following date/time phrase into a JSON object with these fields.
            Do NOT calculate a specific target date or time based on exclusions. Instead, provide the components of the request.

            - "relative_period": string, e.g., "next week", "this week", "tomorrow", "today", or null. This should capture the broader time frame.
            - "specific_date_description": string, a natural language description of a specific date if mentioned (e.g., "June 25th", "Friday", "last workday of the month"), or null.
            - "preferred_hour": integer, 0-23, of the *preferred time*. Assume 9 for morning, 14 for afternoon, 18 for evening, or null.
            - "preferred_minute": integer, 0-59, of the *preferred time*, or null.
            - "excluded_days_of_week": list of integers, 0-6 (Monday=0, Sunday=6), for days explicitly excluded (e.g., "not on Monday" -> [0]), or null.
            - "excluded_time_of_day_phrases": list of strings, for times of day explicitly excluded (e.g., "not in the morning" -> ["morning"]), or null.

            Phrase: '{date_str}'
            Respond ONLY with a single line of valid JSON, no explanation, no markdown, no code block.

            Example for "tomorrow 9 AM": {{"relative_period": "tomorrow", "specific_date_description": null, "preferred_hour": 9, "preferred_minute": 0, "excluded_days_of_week": null, "excluded_time_of_day_phrases": null}}
            Example for "next week but not monday": {{"relative_period": "next week", "specific_date_description": null, "preferred_hour": null, "preferred_minute": null, "excluded_days_of_week": [0], "excluded_time_of_day_phrases": null}}
            Example for "not in the morning on tuesday": {{"relative_period": null, "specific_date_description": "tuesday", "preferred_hour": null, "preferred_minute": null, "excluded_days_of_week": null, "excluded_time_of_day_phrases": ["morning"]}}
            Example for "June 25th 2024 at 3:15 PM": {{"relative_period": null, "specific_date_description": "June 25th 2024", "preferred_hour": 15, "preferred_minute": 15, "excluded_days_of_week": null, "excluded_time_of_day_phrases": null}}
            Example for "last workday of the month": {{"relative_period": null, "specific_date_description": "last workday of the month", "preferred_hour": null, "preferred_minute": null, "excluded_days_of_week": null, "excluded_time_of_day_phrases": null}}
            Example for "next week not monday not in the morning": {{"relative_period": "next week", "specific_date_description": null, "preferred_hour": null, "preferred_minute": null, "excluded_days_of_week": [0], "excluded_time_of_day_phrases": ["morning"]}}
            Example for "two days from now at 5 PM": {{"relative_period": null, "specific_date_description": "two days from now", "preferred_hour": 17, "preferred_minute": 0, "excluded_days_of_week": null, "excluded_time_of_day_phrases": null}}

            Respond with JSON ONLY:
            """
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            import json as _json
            response_text = response.text.strip()
            print(f"[DEBUG] [AI] RAW LLM RESPONSE TEXT: \'\'\'{response_text}\'\'\')")
            
            # Inlining extract_json_from_response logic to bypass AttributeError
            import re as _re
            match = _re.search(r'\{.*?\}', response_text, _re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                json_str = response_text  # fallback: use the whole text

            # Attempt to clean up JSON if LLM adds non-JSON text
            if not json_str.startswith('{'):
                json_str = '{' + json_str.split('{', 1)[-1]
            if not json_str.endswith('}'):
                json_str = json_str.split('}', 1)[0] + '}'

            result = _json.loads(json_str)
             
            print(f"[DEBUG] [AI] RAW LLM PARSED\'{date_str}\': {json.dumps(result, indent=2)}")

            # Validate and normalize parsed output
            parsed_data = {
                "relative_period": result.get("relative_period"),
                "specific_date_description": result.get("specific_date_description"),
                "preferred_hour": result.get("preferred_hour"),
                "preferred_minute": result.get("preferred_minute"),
                "excluded_days_of_week": result.get("excluded_days_of_week"),
                "excluded_time_of_day_phrases": result.get("excluded_time_of_day_phrases"),
            }

            print(f"[DEBUG] [AI] LLM parsed \'{date_str}\' to: {json.dumps(parsed_data, indent=2)}")
            return parsed_data
        except Exception as e:
            print(f"[DEBUG] [AI] LLM failed to parse \'{date_str}\': {e}")
            return None

    def _get_search_date_range(self, ai_parsed_data: Dict, now_ist: datetime) -> Tuple[datetime, datetime]:
        """
        Helper to determine the search date range based on AI parsed data.
        Returns (search_start_dt, search_end_dt).
        For get_events, this should default to today + 14 days if no specific date is provided.
        """
        ist = pytz.timezone('Asia/Kolkata')
        
        # Default search range: start of today to 14 days from now
        search_start_dt = now_ist.replace(hour=0, minute=0, second=0, microsecond=0) 
        search_end_dt = now_ist + timedelta(days=14)

        if ai_parsed_data:
            relative_period = ai_parsed_data.get("relative_period")
            specific_date_description = ai_parsed_data.get("specific_date_description")
            preferred_hour = ai_parsed_data.get("preferred_hour") if ai_parsed_data else None
            preferred_minute = ai_parsed_data.get("preferred_minute")
            if preferred_minute is None:
                preferred_minute = 0
            excluded_days_of_week = ai_parsed_data.get("excluded_days_of_week") or [] if ai_parsed_data else []
            excluded_time_of_day_phrases = ai_parsed_data.get("excluded_time_of_day_phrases") or [] if ai_parsed_data else []

            if relative_period == "tomorrow":
                search_start_dt = now_ist.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                search_end_dt = search_start_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            elif relative_period == "next week":
                # Start of next week (Monday)
                days_until_monday = (7 - now_ist.weekday()) % 7
                if days_until_monday == 0: # If today is Monday, get next Monday
                    days_until_monday = 7
                search_start_dt = (now_ist + timedelta(days=days_until_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
                search_end_dt = search_start_dt + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999) # End of next Sunday
            elif relative_period == "today":
                # For "today", when searching events, we want to search the whole day
                search_start_dt = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
                search_end_dt = now_ist.replace(hour=23, minute=59, second=59, microsecond=999999)
            elif relative_period == "day after tomorrow":
                search_start_dt = now_ist.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=2)
                search_end_dt = search_start_dt.replace(hour=23, minute=59, second=59, microsecond=999999)

            if specific_date_description:
                if specific_date_description.lower() == "last workday of the month":
                    calculated_date = self._get_last_workday_of_month(now_ist)
                    search_start_dt = calculated_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    search_end_dt = calculated_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                elif specific_date_description.lower() == "two days from now":
                    calculated_date = now_ist + timedelta(days=2)
                    search_start_dt = calculated_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    search_end_dt = calculated_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                else:
                    try:
                        parsed_specific_date = parser.parse(specific_date_description, default=now_ist)
                        # Set to start/end of the specific day
                        search_start_dt = parsed_specific_date.replace(hour=0, minute=0, second=0, microsecond=0) 
                        search_end_dt = parsed_specific_date.replace(hour=23, minute=59, second=59, microsecond=999999)

                        # If a preferred hour is given for a specific date, set the start time accordingly,
                        # but DO NOT push it into the future if it's earlier in the day.
                        if preferred_hour is not None:
                            search_start_dt = search_start_dt.replace(hour=preferred_hour, minute=preferred_minute, second=0, microsecond=0)

                    except ValueError as e:
                        print(f"[DEBUG] Could not parse specific_date_description in _get_search_date_range: {specific_date_description}. Error: {e}")

        # For get_events, we want to search the full requested period.
        # The 'future-proofing' logic is primarily for find_next_available.
        # We should ensure start_time is not in the future for get_events unless explicitly specified to be.
        # This final check is only applied if no explicit date/period was parsed by AI,
        # meaning it's a general search for availability.
        if not ai_parsed_data or (not ai_parsed_data.get("relative_period") and not ai_parsed_data.get("specific_date_description")) :
            if search_start_dt < now_ist:
                # For general searches, start from next half-hour if currently in the past
                search_start_dt = now_ist.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=30)
                if search_start_dt.minute not in [0, 30]:
                    search_start_dt = search_start_dt.replace(minute=30 if search_start_dt.minute < 30 else 0, second=0, microsecond=0)
                    if search_start_dt.minute == 0: search_start_dt += timedelta(hours=1)

        return search_start_dt, search_end_dt

    def _find_next_available(self, duration_minutes: int, start_search: datetime = None, date_str: str = None) -> Dict:
        """Find the next available time slot of specified duration (AI date parsing with date_str support)"""
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        
        print(f"[DEBUG] _find_next_available - Input date_str: {date_str}")
        print(f"[DEBUG] _find_next_available - start_search param: {start_search}")
        print(f"[DEBUG] _find_next_available - duration_minutes: {duration_minutes}")
        # Validate duration_minutes
        if duration_minutes is None or not isinstance(duration_minutes, int) or duration_minutes <= 0:
            return {"error": f"Invalid duration_minutes: {duration_minutes}. Please specify a valid meeting duration in minutes."}

        # Initialize variables for search range and AI parsed data
        search_start_dt = None
        search_end_dt = None
        ai_parsed_data = {}
        
        preferred_hour = None
        preferred_minute = 0 # Default to 0
        excluded_days_of_week = []
        excluded_time_of_day_phrases = []

        # First, try to directly parse date_str as a precise datetime
        try:
            if date_str:
                direct_parsed_dt = parser.parse(date_str, default=now_ist)
                # Ensure it's timezone-aware
                if direct_parsed_dt.tzinfo is None:
                    direct_parsed_dt = ist.localize(direct_parsed_dt)
                print(f"[DEBUG] _find_next_available: Successfully directly parsed date_str: {direct_parsed_dt.isoformat()}")
                
                # If direct parsing is successful, use this as the start and the end of that day as the end
                search_start_dt = direct_parsed_dt
                search_end_dt = direct_parsed_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
                
                # Set preferred_hour/minute based on direct parse for subsequent slot finding logic
                preferred_hour = search_start_dt.hour
                preferred_minute = search_start_dt.minute

            else: # If date_str is None or empty, proceed to AI parsing/default behavior
                raise ValueError("date_str is empty, falling back to AI parsing.")

        except (ValueError, TypeError, OverflowError) as e: # Catch parsing errors or explicit fallback
            print(f"[DEBUG] _find_next_available: Direct parsing of '{date_str}' failed ({e}). Falling back to AI parsing.")
            # Fallback to AI parsing for natural language if direct parse fails or date_str is empty
            ai_parsed_data = self._parse_date_with_ai(date_str, now_ist)
            search_start_dt, search_end_dt = self._get_search_date_range(ai_parsed_data, now_ist)
            
            # Recalculate preferred_hour and preferred_minute from AI parsed data
            preferred_hour = ai_parsed_data.get("preferred_hour")
            preferred_minute = ai_parsed_data.get("preferred_minute")
            if preferred_minute is None:
                preferred_minute = 0
            excluded_days_of_week = ai_parsed_data.get("excluded_days_of_week") or []
            excluded_time_of_day_phrases = ai_parsed_data.get("excluded_time_of_day_phrases") or []
            
        # Override search_start_dt if a specific one was passed to invoke (e.g., from date_range_start from older logic)
        # This ensures that if start_search was explicitly provided (e.g., from an older flow), it still takes precedence.
        if start_search and start_search.tzinfo is None: 
            start_search = ist.localize(start_search)
        if start_search and search_start_dt and start_search > search_start_dt:
            search_start_dt = start_search

        # Ensure search starts in the future or current half-hour
        # This logic should still apply to ensure we don't look in the past for availability
        if search_start_dt < now_ist:
            search_start_dt = now_ist.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=30)
            if search_start_dt.minute not in [0, 30]:
                search_start_dt = search_start_dt.replace(minute=30 if search_start_dt.minute < 30 else 0, second=0, microsecond=0)
                if search_start_dt.minute == 0: search_start_dt += timedelta(hours=1)

        print(f"[DEBUG] Final Slot search parameters for user request '{date_str}':")
        print(f"[DEBUG]   Search Start: {search_start_dt.isoformat()} IST")
        print(f"[DEBUG]   Search End:   {search_end_dt.isoformat()} IST")
        print(f"[DEBUG]   Preferred Hour: {preferred_hour}")
        print(f"[DEBUG]   Excluded Days: {excluded_days_of_week}")
        print(f"[DEBUG]   Excluded Times: {excluded_time_of_day_phrases}")

        try:
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=search_start_dt.isoformat(), # Use calculated search_start_dt
                timeMax=search_end_dt.isoformat(),   # Use calculated search_end_dt
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])
            print(f"[DEBUG] Raw API events_result: {json.dumps(events_result, indent=2)}")
            print(f"[DEBUG] Number of events found: {len(events)}")
            
            current_check_time = search_start_dt # Start iterating from the calculated search_start_dt
            duration_delta = timedelta(minutes=duration_minutes)
            
            working_start_hour = 9
            working_end_hour = 18 
            
            available_slots = []
            
            while current_check_time <= search_end_dt and len(available_slots) < 3: # Limit to 3 slots
                # Ensure current_check_time is aligned to a 30-minute mark
                if current_check_time.minute not in [0, 30]:
                    current_check_time = current_check_time.replace(minute=30 if current_check_time.minute < 30 else 0, second=0, microsecond=0)
                    if current_check_time.minute == 0: current_check_time += timedelta(hours=1) # Move to next hour if it was e.g. 1:59 and became 2:00

                # 1. Apply day-of-week exclusions
                if current_check_time.weekday() in excluded_days_of_week:
                    print(f"[DEBUG] Skipping excluded day ({current_check_time.strftime('%A')}): {current_check_time.isoformat()}")
                    current_check_time = (current_check_time + timedelta(days=1)).replace(hour=working_start_hour, minute=0, second=0, microsecond=0)
                    continue # Skip to next day
                
                # 2. Apply time-of-day exclusions
                current_hour = current_check_time.hour
                skip_due_to_time_exclusion = False
                for phrase in excluded_time_of_day_phrases:
                    if phrase == "morning" and 6 <= current_hour < 12:
                        skip_due_to_time_exclusion = True
                        current_check_time = current_check_time.replace(hour=12, minute=0, second=0, microsecond=0) # Jump to noon
                        break
                    elif phrase == "afternoon" and 12 <= current_hour < 17:
                        skip_due_to_time_exclusion = True
                        current_check_time = current_check_time.replace(hour=17, minute=0, second=0, microsecond=0) # Jump to evening
                        break
                    elif phrase == "evening" and 17 <= current_hour < 21:
                        skip_due_to_time_exclusion = True
                        current_check_time = current_check_time.replace(hour=21, minute=0, second=0, microsecond=0) # Jump to night
                        break
                    elif phrase == "night" and (21 <= current_hour or current_hour < 6):
                        skip_due_to_time_exclusion = True
                        current_check_time = (current_check_time + timedelta(days=1)).replace(hour=working_start_hour, minute=0, second=0, microsecond=0) # Jump to next morning
                        break
                if skip_due_to_time_exclusion:
                    print(f"[DEBUG] Skipping excluded time ('{phrase}'): {current_check_time.isoformat()}")
                    continue

                # 3. Apply general working hours and preferred hour (if applicable)
                effective_start_hour_for_day = working_start_hour
                # If a preferred hour is specified, and it's within current day, try to align to it
                if preferred_hour is not None and current_check_time.date() == search_start_dt.date():
                    if current_check_time.hour < preferred_hour:
                        current_check_time = current_check_time.replace(hour=preferred_hour, minute=preferred_minute, second=0, microsecond=0)
                        if current_check_time < now_ist: # Don't start in the past if current day
                             current_check_time = now_ist.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=30)
                             if current_check_time.minute not in [0, 30]:
                                current_check_time = current_check_time.replace(minute=30 if current_check_time.minute < 30 else 0, second=0, microsecond=0)
                                if current_check_time.minute == 0: current_check_time += timedelta(hours=1)

                # Ensure current_check_time is within daily working hours for the *current day* being checked
                if current_check_time.hour < effective_start_hour_for_day:
                    current_check_time = current_check_time.replace(hour=effective_start_hour_for_day, minute=0, second=0, microsecond=0)
                
                proposed_end = current_check_time + duration_delta
                
                # Check if the proposed slot goes beyond the daily working end hour or the overall search_end_dt
                # If proposed_end.hour is equal to working_end_hour, it's fine if proposed_end.minute is 0 (i.e. slot ends exactly at working_end_hour).
                if (proposed_end.date() == current_check_time.date() and proposed_end.hour > working_end_hour) or \
                   (proposed_end.date() == current_check_time.date() and proposed_end.hour == working_end_hour and proposed_end.minute > 0) or \
                   proposed_end > search_end_dt: # Check against overall search end
                    print(f"[DEBUG] Proposed_end ({proposed_end.isoformat()}) goes past daily/overall limits. Moving to next day.")
                    current_check_time = (current_check_time + timedelta(days=1)).replace(hour=working_start_hour, minute=0, second=0, microsecond=0) # Jump to next day's start
                    continue

                conflicted = False
                for event in events:
                    event_start = parser.parse(event['start'].get('dateTime', event['start'].get('date')))
                    event_end = parser.parse(event['end'].get('dateTime', event['end'].get('date')))
                    
                    # Ensure event times are in the correct timezone for comparison
                    if event_start.tzinfo:
                        event_start = event_start.astimezone(ist)
                    else:
                        event_start = ist.localize(event_start)
                    if event_end.tzinfo:
                        event_end = event_end.astimezone(ist)
                    else:
                        event_end = ist.localize(event_end)
                    
                    # Check for conflict: Proposed slot overlaps with an event
                    if (current_check_time < event_end and proposed_end > event_start):
                        conflicted = True
                        print(f"[DEBUG] Conflict detected with event: {event.get('summary', 'No Title')} from {event_start.isoformat()} to {event_end.isoformat()}")
                        # Move current_check_time past the conflicting event, aligned to next 30 min mark
                        current_check_time = event_end
                        if current_check_time.minute not in [0, 30]:
                            current_check_time = current_check_time.replace(minute=30 if current_check_time.minute < 30 else 0, second=0, microsecond=0)
                            if current_check_time.minute == 0: current_check_time += timedelta(hours=1)
                        
                        # After advancing past conflict, ensure it's still within daily limits. If not, jump to next day.
                        if current_check_time.hour >= working_end_hour: # If moved past end of day
                            current_check_time = (current_check_time + timedelta(days=1)).replace(hour=working_start_hour, minute=0, second=0, microsecond=0)
                            if current_check_time.minute not in [0, 30]:
                                current_check_time = current_check_time.replace(minute=30 if current_check_time.minute < 30 else 0, second=0, microsecond=0)
                                if current_check_time.minute == 0: current_check_time += timedelta(hours=1)
                        break
                
                if conflicted:
                    print(f"[DEBUG] Conflicted. Continuing to next iteration with current_check_time={current_check_time.isoformat()}")
                    continue # Re-evaluate current_check_time after conflict

                # If no conflict and slot is valid
                print(f"[DEBUG] Slot found and added: {current_check_time.isoformat()} to {proposed_end.isoformat()}")
                available_slots.append({
                    "start_time": current_check_time.isoformat(),
                    "end_time": proposed_end.isoformat(),
                    "formatted_time": f"{current_check_time.strftime('%A, %B %d at %I:%M %p')} to {proposed_end.strftime('%I:%M %p')}"
                })
                
                # Move to the next 30-minute interval for the next slot check
                current_check_time += timedelta(minutes=30)
                
            print(f"[DEBUG] Final available_slots count: {len(available_slots)}")
            if available_slots:
                primary_slot = available_slots[0]
                return {
                    "available": True,
                    "start_time": primary_slot["start_time"],
                    "end_time": primary_slot["end_time"],
                    "message": f"Found {len(available_slots)} available {duration_minutes}-minute slots",
                    "all_slots": available_slots,
                    "days_ahead": (available_slots[0]["start_time"] and (parser.parse(available_slots[0]["start_time"]).date() - now_ist.date()).days) or 0,
                    "preferred_hour": preferred_hour # Pass preferred_hour back in results for debug
                }
            else:
                return {
                    "available": False,
                    "message": f"No available {duration_minutes}-minute slots found in the selected period.",
                    "days_ahead": 0, # If no slots, days_ahead is not relevant
                    "preferred_hour": preferred_hour # Pass preferred_hour back in results for debug
                }
        except Exception as e:
            print(f"[DEBUG] HttpError in _find_next_available: {e}")
            return {"error": f"An error occurred while searching: {e}"}
    
    def invoke(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute the calendar tool action"""
        try:
            if action == "check_availability":
                if not kwargs.get('start_time') or not kwargs.get('end_time'):
                    return {"error": "start_time and end_time are required for checking availability."}
                start_dt = self._parse_time(kwargs['start_time'])
                end_dt = self._parse_time(kwargs['end_time'])
                return self._check_availability(start_dt, end_dt)
            elif action == "create_meeting":
                if not kwargs.get('start_time') or not kwargs.get('end_time') or not kwargs.get('title'):
                    return {"error": "start_time, end_time, and title are required for creating a meeting."}
                start_dt = self._parse_time(kwargs['start_time'])
                end_dt = self._parse_time(kwargs['end_time'])
                return self._create_meeting(start_dt, end_dt, kwargs['title'], kwargs.get('description', ''))
            elif action == "find_next_available":
                if not kwargs.get('duration_minutes'):
                    return {"error": "duration_minutes is required for finding next available slot."}
                # Support for specific day or relative day
                date_str = None
                if kwargs.get('preferred_time'):
                    date_str = kwargs['preferred_time']
                elif kwargs.get('date_str'):
                    date_str = kwargs['date_str']
                start_search = None
                if kwargs.get('date_range_start'):
                    start_search = self._parse_time(kwargs['date_range_start'])
                print(f"[DEBUG] Tool call dict: {kwargs}")
                
                return self._find_next_available(
                    kwargs['duration_minutes'],
                    start_search,
                    date_str
                )
            elif action == "get_events":
                if not kwargs.get('query'):
                    return {"error": "query is required for getting events."}
                search_start = None
                search_end = None
                if kwargs.get('date_range_start'):
                    search_start = self._parse_time(kwargs['date_range_start'])
                if kwargs.get('date_range_end'):
                    search_end = self._parse_time(kwargs['date_range_end'])
                return self._get_events(kwargs['query'], search_start, search_end)
            else:
                return {"error": f"Unknown action '{action}'. Available actions: check_availability, create_meeting, find_next_available"}
        except Exception as e:
            return {"error": str(e)}

    def _format_iso_to_natural(self, iso_time_str: str) -> str:
        """Helper to format ISO timestamp to natural language."""
        try:
            dt_object = datetime.fromisoformat(iso_time_str)
            return dt_object.strftime("%A, %B %d at %I:%M %p")
        except ValueError:
            return iso_time_str # Return as is if parsing fails

    def extract_json_from_response(self, text):
        # Try to find the first {...} JSON object in the text
        import re
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text  # fallback: return the whole text 