import os
import json
import config
import google.generativeai as genai
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from datetime import datetime
import pytz
import tools # Import the tools module

# Load environment variables
load_dotenv()

# VAPI Configuration (will be used later)
VAPI_API_KEY = os.getenv('VAPI_API_KEY')

@dataclass
class VAPIIntegration:
    """Simple VAPI integration class that can be expanded later"""
    api_key: str
    base_url: str = "https://api.vapi.ai"
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if VAPI is properly configured."""
        return bool(VAPI_API_KEY)
    
    async def start_call(self, phone_number: str, assistant_id: str) -> Dict[str, Any]:
        """Start a voice call using VAPI."""
        # Implementation will be added when VAPI is integrated
        return {"status": "VAPI integration not yet implemented"}
    
    async def handle_voice_input(self, audio_data: bytes) -> Dict[str, Any]:
        """Process voice input and return the assistant's response."""
        # Implementation will be added when VAPI is integrated
        return {"status": "VAPI integration not yet implemented"}

load_dotenv()


def get_llm() -> genai.GenerativeModel:
    """Initialize and return the Gemini LLM with basic configuration."""
    if not config.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    genai.configure(api_key=config.GOOGLE_API_KEY)

    # Get current date in IST
    ist = pytz.timezone('Asia/Kolkata')
    current_date = datetime.now(ist).strftime('%Y-%m-%d')

    # Format the system prompt with the current date
    formatted_system_prompt = config.SYSTEM_PROMPT.format(current_date=current_date)

    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 1024,
        },
        system_instruction=formatted_system_prompt,
        tools=[
            tools.parse_natural_date,
            tools.get_free_slots,
            tools.book_meeting_tool,
            tools.get_events_tool
        ]
    )