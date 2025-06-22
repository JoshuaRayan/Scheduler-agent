import json
import pytz
import google_calendar
from parsedatetime import parsedatetime
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from google_calendar import GoogleCalendarService

# Initialize the parser
cal = parsedatetime.Calendar()

def parse_natural_date(text: str, relative_to: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Parse natural language date/time strings into ISO format, providing more detail.
    
    Examples:
        "tomorrow at 3pm" -> {"datetime_iso": "2023-06-23T15:00:00+05:30", "date_iso": "2023-06-23", "time_iso": "15:00:00", "time_specified": True}
        "next monday" -> {"datetime_iso": "2023-06-26T00:00:00+05:30", "date_iso": "2023-06-26", "time_iso": "00:00:00", "time_specified": False}
        
    Args:
        text: Natural language date/time string
        relative_to: Reference datetime (defaults to now in IST)
        
    Returns:
        Dict with 'datetime_iso' (ISO 8601 string), 'date_iso', 'time_iso', 'time_specified' (boolean), or 'error' if parsing failed
    """
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = relative_to or datetime.now(ist)
        
        # Use parseDT for more detailed parsing results, including whether time was specified
        time_struct, status = cal.parseDT(text, sourceTime=now)
        
        if not status: # If parsing failed, status will be 0
            return {"error": f"Could not parse date/time: {text}"}
            
        dt = time_struct # parseDT directly returns a datetime object if successful
        if dt.tzinfo is None:
            dt = ist.localize(dt)
        else:
            dt = dt.astimezone(ist) # Ensure it's in IST if timezone info exists
        
        result = {"datetime_iso": dt.isoformat()}
        
        # Check if time was explicitly mentioned (status & 2 means time was parsed)
        result["time_specified"] = bool(status & 2) 
        
        # Add date_iso and time_iso for more granularity
        result["date_iso"] = dt.strftime('%Y-%m-%d')
        result["time_iso"] = dt.strftime('%H:%M:%S')

        return result
        
    except Exception as e:
        return {"error": f"Error parsing date: {str(e)}"}

def format_duration(minutes: int) -> str:
    """Convert minutes to a human-readable duration string."""
    hours, mins = divmod(minutes, 60)
    if hours > 0 and mins > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} {mins} minute{'s' if mins > 1 else ''}"
    elif hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''}"
    return f"{mins} minute{'s' if mins > 1 else ''}"


def get_free_slots(date: str, duration_minutes: int = 60) -> dict:
    """
    Find available free time slots for a meeting on the specified date.

    Args:
        date (str): Date in 'YYYY-MM-DD' format.
        duration_minutes (int): Duration of the meeting in minutes.

    Returns:
        dict: Dictionary containing:
            - free_slots: List of dicts with 'start' and 'end' in ISO format
            - formatted_slots: List of human-readable time slots
            - error: Error message if any
    """
    try:
        # Get the raw datetime slots
        datetime_slots = google_calendar.find_free_slots(date, duration_minutes)
        
        # Format slots for display
        formatted_slots = google_calendar.format_slots(datetime_slots)
        
        # Prepare the response with both datetime and formatted versions
        response = {
            'free_slots': [
                {
                    'start': start.isoformat(),
                    'end': end.isoformat(),
                    'formatted': f"{start.strftime('%I:%M %p')} - {end.strftime('%I:%M %p')}"
                }
                for start, end in datetime_slots
            ],
            'formatted_slots': formatted_slots,
            'date': date,
            'duration_minutes': duration_minutes
        }
        
        return response
        
    except Exception as e:
        error_msg = f"Error finding free slots: {str(e)}"
        return {"free_slots": [], "formatted_slots": [], "error": error_msg}

def book_meeting_tool(start_time: str, end_time: str, title: str) -> Dict:
    """
    Books a meeting in the user's Google Calendar.
    
    Args:
        start_time: The start time of the meeting in ISO format (e.g., "YYYY-MM-DDTHH:MM:SS+05:30").
        end_time: The end time of the meeting in ISO format (e.g., "YYYY-MM-DDTHH:MM:SS+05:30").
        title: The title of the meeting.

    Returns:
        A dictionary with keys:
        - "event_link": string, URL to the created event (if successful).
        - "error": string (if any)
    """
    try:
        ist = pytz.timezone('Asia/Kolkata')
        start_dt = datetime.fromisoformat(start_time).astimezone(ist)
        end_dt = datetime.fromisoformat(end_time).astimezone(ist)

        event_link = google_calendar.book_meeting(start_dt, end_dt, title)
        
        if event_link:
            return {
                "event_link": event_link
            }
        else:
            return {"success": False, "message": "Failed to book meeting.", "error": "Unknown booking error"}

    except Exception as e:
        return {"success": False, "message": f"An error occurred while booking meeting: {e}", "error": str(e)}


def get_events_tool(date: str) -> Dict:
    """
    Retrieves all busy events for a specific date from the user's calendar.

    Args:
        date (str): The date in 'YYYY-MM-DD' format to retrieve events for.

    Returns:
        A dictionary with keys:
        - "events": List of dictionaries, each with "start" (ISO format) and "end" (ISO format).
        - "error": string (if any)
    """
    try:
        busy_slots_raw = google_calendar.get_busy_events_for_day(date)

        events_list = [{
            "start": slot[0].isoformat(),
            "end": slot[1].isoformat()
        } for slot in busy_slots_raw]

        return {
            "events": events_list
        }

    except Exception as e:
        return {"events": [], "error": str(e)}

