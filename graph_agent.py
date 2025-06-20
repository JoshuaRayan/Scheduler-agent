from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
import json
import re
import google.generativeai as genai
import config
from datetime import datetime, timedelta
import pytz
from dateutil import parser

from calendar_tool import CalendarTool

@dataclass
class Message:
    role: str  # "user", "assistant", or "system"
    content: str

class AgentState(TypedDict, total=False):
    messages: List[Message]
    user_input: str
    agent_response: str
    tool_calls: List[Dict]
    conversation_context: Dict[str, Any]
    iteration_count: int
    tool_results: List[Dict]
    event_search_result: Optional[Dict]

class SmartSchedulerAgent:
    def __init__(self):
        """Initialize the agent with Google's Gemini model"""
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize calendar tool with error handling
        try:
            print("\n=== Initializing Calendar Tool ===")
            self.calendar_tool = CalendarTool()
            if self.calendar_tool.service: # Check if service was actually built
                print("Calendar tool initialized successfully")
            else:
                print("Calendar tool initialized but service not ready (token.json likely missing). Will attempt re-auth later.")
        except FileNotFoundError as e:
            print(f"Warning: Calendar tool initialization failed: {e}. Authentication required.")
            self.calendar_tool = CalendarTool() # Still create the object, but service will be None
            self.calendar_tool.service = None # Explicitly ensure service is None
        except Exception as e:
            print(f"Warning: Calendar tool initialization failed: {e}")
            self.calendar_tool = None
        
        self.graph = self._create_graph()
        print("=== End Calendar Tool Initialization ===\n")
    
    def ensure_calendar_authenticated(self):
        """Attempts to re-authenticate the calendar tool if its service is not available."""
        if not self.calendar_tool or not self.calendar_tool.service:
            print("Attempting to re-authenticate Calendar Tool...")
            try:
                # If CalendarTool object exists but service is None, try re-authenticating it
                if self.calendar_tool:
                    self.calendar_tool._authenticate()
                else:
                    # If CalendarTool object itself is None, create a new one and authenticate
                    self.calendar_tool = CalendarTool()
                
                if self.calendar_tool and self.calendar_tool.service:
                    print("Calendar Tool re-authenticated successfully.")
                else:
                    print("Calendar Tool re-authentication attempt failed.")
            except Exception as e:
                print(f"Error during Calendar Tool re-authentication: {e}")
                self.calendar_tool.service = None # Ensure service is None on failure
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            }
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def _process_initial_state(self, state: AgentState) -> bool:
        """
        Processes the initial state for the agent node, including iteration count,
        context initialization, and message preparation.

        Args:
            state: The current AgentState.

        Returns:
            bool: True if processing should continue, False if iteration limit reached.
        """
        messages = state.get("messages", [])
        user_input = state.get("user_input", "")
        conversation_context = state.get("conversation_context", {})
        tool_results = state.get("tool_results", [])
        iteration_count = state.get("iteration_count", 0)
        
        print(f"[DEBUG] _agent_node START - context['step']: {conversation_context.get('step')}, context['duration']: {conversation_context.get('duration')}, context['preferred_time']: {conversation_context.get('preferred_time')}, context['event_to_follow_up']: {conversation_context.get('event_to_follow_up')}")

        # Increment iteration count
        iteration_count += 1
        state["iteration_count"] = iteration_count
        
        # Prevent infinite loops (This check is maintained to prevent runaway costs, but response logic will be prioritized)
        if iteration_count > 5:
            print(f"=== Stopping due to iteration limit ({iteration_count}) ===")
            # If we stop due to limit, try to provide a meaningful response if tool results are available
            if tool_results:
                state["agent_response"] = self._generate_response_from_tool_results(tool_results, conversation_context)
            else:
                state["agent_response"] = "I apologize, but I'm having trouble processing your request. Please try again with a simpler request."
                state["tool_calls"] = []
                state["tool_results"] = [] # Clear results to prevent re-processing on next turn if any
                return False # Indicate that processing should stop
        
        print("\n=== Agent Node Debug ===")
        print(f"Iteration: {iteration_count}")
        print(f"User Input: {user_input}")
        print(f"Conversation Context: {json.dumps(conversation_context, indent=2)}")
        print(f"[DEBUG] _agent_node - Received conversation_context at start: {json.dumps(conversation_context, indent=2)}")
        print(f"Tool Results: {json.dumps(tool_results, indent=2)}")
        
        # Initialize conversation context if not present
        if not conversation_context:
            conversation_context = {
                "step": "initial",
                "duration": None,
                "preferred_time": None,
                "title": None,
                "available_slots": [],
                "last_action": None,
                "parsed_input": None
            }
            state["conversation_context"] = conversation_context # Update state explicitly
        
        # Add user input if provided
        if user_input:
            messages.append(Message(role="user", content=user_input))
            conversation_context["parsed_input"] = user_input

        # Add tool results to messages if available (for LLM to see on its next pass)
        if tool_results:
            tool_result_text = "Tool results:\n" + "\n".join(
                json.dumps(result, indent=2) for result in tool_results
            )
            messages.append(Message(role="system", content=tool_result_text))

        # Add system message if not present
        if not messages or messages[0].role != "system":
            system_prompt = f"""{config.SYSTEM_PROMPT}

IMPORTANT: You have access to a calendar tool. You MUST use TOOL_CALL to check or book meetings.
NEVER say you cannot access the calendar. If you do not have enough information, ask for it, but do NOT say you cannot access the calendar.

NEGATIVE EXAMPLES (DO NOT DO THIS):
- 'I can't access your calendar'
- 'You'll need to book this yourself'
- 'I am unable to book meetings for you'

POSITIVE EXAMPLES:
- TOOL_CALL: {{"action": "find_next_available", "duration_minutes": 60}}
- TOOL_CALL: {{"action": "find_next_available", "duration_minutes": 120, "date_str": "next week but not monday"}} # Use date_str for all nuances, not date_range_start/end
- TOOL_CALL: {{"action": "create_meeting", ...}}

You have access to a calendar tool with these actions:
- find_next_available: Find available time slots (requires duration_minutes as integer, optionally date_str for natural language date/time)
- create_meeting: Book a meeting (requires start_time, end_time, title)

CRITICAL RULES - READ CAREFULLY:
1. When you need to use a tool, you MUST respond with: TOOL_CALL: {{"action": "action_name", "param1": "value1"}}
2. If the user wants different slots (mentions new time, duration, or says "different time"), call find_next_available again
3. If available_slots exist but user wants changes, clear them and search for new ones
4. Only call create_meeting when the user explicitly confirms they want to book
5. After calling find_next_available and getting results, STOP and present the slots to the user
6. Meeting title is OPTIONAL. If not provided, use the default title 'Meeting'. Only ask for a title if the user is booking and hasn't provided one.
7. If the user mentions a specific time (like "Friday at 10 AM"), clear any existing slots and search for new ones
8. IMPORTANT for find_next_available: If conversation_context.preferred_time is set, always pass it as "date_str" in the tool call. DO NOT include "date_range_start" or "date_range_end" as parameters if a preferred_time is available, as the system will handle complex date parsing via "date_str".

CONVERSATION FLOW:
1. User provides duration and time → Call find_next_available
2. Present available slots to user
3. If user wants different slots → Call find_next_available again
4. Ask for meeting title if not provided and user is booking
5. User confirms booking → Call create_meeting

**IMPORTANT: If user wants different slots, DO call find_next_available again to get new options.**

CRITICAL: For create_meeting, use ISO format timestamps (YYYY-MM-DDTHH:MM:SS), NOT relative time strings like "tomorrow 5:00 PM"

Current conversation context: {json.dumps(conversation_context, indent=2)}

Remember: Be conversational and helpful. Use tools when you need to check calendar or book meetings.
When you find available slots, present them to the user in a friendly way and ask if they want to book."""
            messages.insert(0, Message(role="system", content=system_prompt))
            state["messages"] = messages # Update state explicitly

        return True # Indicate that processing should continue

    def _agent_node(self, state: AgentState) -> AgentState:
        """Process the agent node using LLM to generate responses and tool calls"""
        if not self._process_initial_state(state):
            return state # Return early if iteration limit was reached or other early exit condition

        # Re-extract state variables after initial processing, as they might have been updated
        messages = state["messages"]
        user_input = state["user_input"]
        conversation_context = state["conversation_context"]
        tool_results = state["tool_results"]
        iteration_count = state["iteration_count"]

        try:
            # Update conversation context based on user input using LLM parsing
            parsed_context_data = self._parse_context_with_llm(user_input, conversation_context)
            conversation_context.update(parsed_context_data)
            
            # After parsing, intelligently merge preferred_time if a refinement occurred
            original_preferred_time = state.get("conversation_context", {}).get("preferred_time")
            newly_parsed_preferred_time = conversation_context.get("preferred_time")

            if original_preferred_time and newly_parsed_preferred_time and original_preferred_time != newly_parsed_preferred_time:
                is_original_concrete = False
                try:
                    parser.parse(original_preferred_time) # Just to check if it's parseable as a date
                    is_original_concrete = True
                except (ValueError, TypeError):
                    pass # Not a concrete date string

                time_of_day_phrases = ["morning", "afternoon", "evening", "night"]
                is_new_time_of_day = any(phrase in newly_parsed_preferred_time.lower() for phrase in time_of_day_phrases)
                
                if is_original_concrete and is_new_time_of_day:
                    try:
                        original_dt = parser.parse(original_preferred_time)
                        now_ist = datetime.now(pytz.timezone('Asia/Kolkata')) # Use current time for context for _parse_date_with_ai
                        refined_ai_data = self.calendar_tool._parse_date_with_ai(newly_parsed_preferred_time, original_dt) # Pass original_dt as context
                        
                        if refined_ai_data and refined_ai_data.get("preferred_hour") is not None:
                            refined_dt = original_dt.replace(
                                hour=refined_ai_data["preferred_hour"],
                                minute=refined_ai_data.get("preferred_minute", 0)
                            )
                            conversation_context["preferred_time"] = refined_dt.strftime('%Y-%m-%d %I:%M %p')
                            print(f"[DEBUG] Agent Node: Refined preferred_time to {conversation_context['preferred_time']} by combining date and new time-of-day.")
                    except Exception as e:
                        print(f"[DEBUG] Agent Node: Error refining preferred_time: {e}")

            self._last_user_input = user_input

            # Initialize determined variables for this turn
            final_tool_calls = []
            final_clean_response = ""
            
            # --- NEW LOGIC: Prioritize processing event_search_result first ---
            # This block handles the case where get_events tool has just run
            if state.get("event_search_result"):
                event_search_result = state["event_search_result"]
                state["event_search_result"] = None # Clear it immediately after using, to prevent re-processing

                if event_search_result.get("success") and event_search_result.get("events"):
                    found_events = event_search_result["events"]
                    # Assume we only care about the first event for scheduling after
                    event_end = found_events[0]["end"]
                    
                    from dateutil import parser as dtparser
                    from datetime import timedelta
                    event_end_dt = dtparser.parse(event_end)
                    
                    # Use follow_up_offset_days from conversation_context, default to 0 if not specified
                    days_after_event = conversation_context.get("follow_up_offset_days", 0) 
                    slot_start_dt = event_end_dt + timedelta(days=days_after_event)
                    slot_start_str = slot_start_dt.strftime('%Y-%m-%d %I:%M %p')
                    
                    duration = conversation_context.get("duration")

                    if duration: # Ensure we have a duration before chaining to find_next_available
                        final_tool_calls = [{
                            "action": "find_next_available",
                            "duration_minutes": duration,
                            "date_str": slot_start_str
                        }]
                        # Update conversation_context with the concrete date for future turns/overrides
                        conversation_context["preferred_time"] = slot_start_str
                        final_clean_response = f"Okay, I found '{found_events[0]['summary']}'. Now searching for available slots for {duration} minutes starting after {self.calendar_tool._format_iso_to_natural(event_end)}."
                        conversation_context["step"] = "find_slots" # Set step to find_slots for the next iteration
                        print(f"[DEBUG] Agent Node: Chaining to find_next_available after event search: {final_tool_calls}")
                    else:
                        # If no duration, ask for it
                        final_tool_calls = [] # No tool call yet
                        final_clean_response = f"I found '{found_events[0]['summary']}'. What duration would you like for the meeting after this event?"
                        conversation_context["step"] = "gathering_duration"
                        print("[DEBUG] Agent Node: Found event, but no duration. Asking for duration.")
                else:
                    # No event found or success was false in event_search_result
                    final_tool_calls = [] # No tool call
                    final_clean_response = f"I couldn't find an event matching '{conversation_context.get('event_to_follow_up', 'your query')}'. Could you please provide more details or try a different event name?"
                    conversation_context["event_to_follow_up"] = None
                    conversation_context["step"] = "initial"
                
                # After processing event_search_result, update state and return to let the graph continue
                state["messages"] = messages + [Message(role="assistant", content=final_clean_response)] # Add response for current turn
                state["agent_response"] = final_clean_response
                state["conversation_context"] = conversation_context
                state["tool_calls"] = final_tool_calls
                state["tool_results"] = [] # Clear tool results, as this path is for event_search_result
                return state # IMPORTANT: End agent node execution here to allow graph to transition

            # --- IMPORTANT: Prioritize generating clean_response from tool_results if available ---
            # This block runs AFTER any other logic that sets final_clean_response, ensuring tool results are paramount.
            if tool_results:
                final_clean_response = self._generate_response_from_tool_results(tool_results, conversation_context)
                
                # Check the action of the tool result to determine next step more accurately
                first_tool_result = tool_results[0] if tool_results else None
                if first_tool_result and first_tool_result.get("action") == "find_next_available":
                    if conversation_context.get("available_slots"): # If slots were found by find_next_available
                        conversation_context["step"] = "confirm_booking"
                        print("[DEBUG] Agent Node: Set step to 'confirm_booking' after finding slots.")
                    else: # If no slots were found by find_next_available
                        conversation_context["step"] = "initial"
                        print("[DEBUG] Agent Node: Set step to 'initial' after no slots found by find_next_available.")
                
                # If tool_results were processed, we should not proceed to generate new LLM calls.
                # Set final_tool_calls to empty here and then return state, unless a new tool call was explicitly set by a chaining logic above.
                # However, for tool_results, the chain is usually complete on this turn for the agent.
                final_tool_calls = [] # Ensure no new tool calls are made after processing results

                state["messages"] = messages + [Message(role="assistant", content=final_clean_response)]
                state["agent_response"] = final_clean_response
                state["conversation_context"] = conversation_context
                state["tool_calls"] = final_tool_calls # This will be empty after processing results
                state["tool_results"] = [] # Clear results to prevent re-processing on next turn if any
                return state # IMPORTANT: End agent node execution here if tool results were processed

            # --- NEW: Logic to initiate get_events when in find_event_then_slots step ---
            # This block will be executed if no immediate tool_results or event_search_result were processed
            # and the conversation is specifically in the state of needing to find an event first.
            current_step = conversation_context.get("step") # Re-read current_step after potential modifications
            if current_step == "find_event_then_slots":
                event_query = conversation_context.get("event_to_follow_up")
                # Add condition: only call get_events if event_search_result is not yet processed (is None)
                # AND if preferred_time is not yet a concrete date (meaning we haven't found the follow-up date yet)
                preferred_time_is_concrete = False
                try:
                    if conversation_context.get("preferred_time"):
                        parser.parse(conversation_context["preferred_time"])
                        preferred_time_is_concrete = True
                except (ValueError, TypeError):
                    pass # Not a concrete date string

                if event_query and not state.get("event_search_result") and not preferred_time_is_concrete:
                    print(f"[DEBUG] Agent Node: Initiating event search for '{event_query}'.")
                    tool_call_params = { "action": "get_events", "query": event_query }

                    # Determine the search range for get_events
                    preferred_time_for_event_search = conversation_context.get("preferred_time")
                    now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))

                    event_search_start = None
                    event_search_end = None

                    if preferred_time_for_event_search:
                        # For initial event search, allow _parse_date_with_ai to handle natural language
                        ai_parsed_data_for_event = self.calendar_tool._parse_date_with_ai(preferred_time_for_event_search, now_ist)
                        if ai_parsed_data_for_event:
                            event_search_start, event_search_end = self.calendar_tool._get_search_date_range(ai_parsed_data_for_event, now_ist)
                            print(f"[DEBUG] Agent Node: Added time range to get_events (parsed): {event_search_start.isoformat()} to {event_search_end.isoformat()}")
                    
                    # If no specific time was parsed, default to today + 14 days
                    if not event_search_start or not event_search_end:
                        event_search_start = now_ist.replace(hour=0, minute=0, second=0, microsecond=0) # Start of today
                        event_search_end = now_ist + timedelta(days=14) # Next 14 days
                        print(f"[DEBUG] Agent Node: Added time range to get_events (default 14 days): {event_search_start.isoformat()} to {event_search_end.isoformat()}")

                    tool_call_params["start_time"] = event_search_start.isoformat()
                    tool_call_params["end_time"] = event_search_end.isoformat()

                    final_tool_calls = [tool_call_params]
                    final_clean_response = f"Okay, let me find details about '{event_query}' first."
                    conversation_context["step"] = "awaiting_event_results_for_scheduling"
                else:
                    print("[DEBUG] Agent Node: find_event_then_slots step but either no event_to_follow_up, event_search_result already exists, or preferred_time is already concrete.")
                    final_tool_calls = [] # No tool call if missing event query, or event already processed
                    # If preferred_time is already concrete and duration is present, we should transition to find_slots
                    if preferred_time_is_concrete and conversation_context.get("duration"):
                        conversation_context["step"] = "find_slots"
                        final_clean_response = "Okay, I'm ready to find slots based on the updated time."
                    else:
                        final_clean_response = "I'm set to find an event, but I don't know which one. Could you please specify?"
                        conversation_context["step"] = "initial"

                # After initiating the get_events tool call, we update state and return
                state["messages"] = messages + [Message(role="assistant", content=final_clean_response)]
                state["agent_response"] = final_clean_response
                state["conversation_context"] = conversation_context
                state["tool_calls"] = final_tool_calls
                return state # Crucial to allow graph to transition to tools

            # --- Other conversational prompts/LLM interaction (fallback if no specific step logic handled) ---
            # This section now acts as the general LLM planner if specific tool-related steps aren't met.
            # We need to make sure this doesn't override `final_tool_calls` if they were set above.
            if not final_tool_calls: # Only call LLM if no tool calls explicitly set by step logic
                # Convert messages to Gemini format
                gemini_messages = []
                for msg in messages:
                    if msg.role == "system":
                        if gemini_messages:
                            gemini_messages[0]["parts"][0]["text"] = msg.content + "\n\n" + gemini_messages[0]["parts"][0]["text"]
                        else:
                            gemini_messages.append({"role": "user", "parts": [{"text": msg.content}]})
                    else:
                        gemini_messages.append({"role": msg.role, "parts": [{"text": msg.content}]})

                # Generate response using Gemini
                response = self.model.generate_content(gemini_messages)
                response_content = response.text
                print(f"LLM Response: {response_content}")
                
                # This part will now be primarily for extracting tool calls if LLM makes an unprompted one
                # or if the initial planning was not caught by specific step logic.
                final_tool_calls = self._extract_tool_calls(response_content, conversation_context, user_input)
                if not final_clean_response.strip(): # Only update if not set by specific step logic
                    final_clean_response = self._clean_response_content(response_content)
            

            print(f"Final determined tool_calls: {json.dumps(final_tool_calls, indent=2)}")

            # Fallback for when clean_response might still be empty (e.g., LLM only returned tool calls)
            if not final_clean_response.strip() and final_tool_calls:
                if final_tool_calls[0].get("action") == "find_next_available":
                    final_clean_response = "Okay, searching for available slots based on your request."
                elif final_tool_calls[0].get("action") == "create_meeting":
                    final_clean_response = "Okay, I'm booking that meeting for you."
                elif final_tool_calls[0].get("action") == "get_events":
                    final_clean_response = "Okay, I'm looking for those events."
                else:
                    final_clean_response = "Processing your request."
            
            # Ensure no tool calls if in an information gathering state, regardless of LLM output
            if conversation_context.get("step") in ["gathering_duration", "gathering_time", "need_title"]:
                print(f"[DEBUG] _agent_node: Clearing tool_calls because step is {conversation_context.get('step')}")
                final_tool_calls = []

            # If we're in find_slots, have duration and preferred_time, and no tool call is pending, issue the tool call
            # This handles cases where the LLM might not produce a TOOL_CALL in its raw response, but the context indicates it's needed.
            if (conversation_context.get("step") == "find_slots"
                and conversation_context.get("duration")
                and conversation_context.get("preferred_time")
                and not final_tool_calls # Only generate if no other tool calls have been set
                and not conversation_context.get("available_slots", [])):
                final_tool_calls = [{
                    "action": "find_next_available",
                    "duration_minutes": conversation_context["duration"],
                    "date_str": conversation_context["preferred_time"]
                }]
                final_clean_response = "Searching for available slots..."

            # Update state (this block is now the final state update before returning)
            state["messages"] = messages + [Message(role="assistant", content=final_clean_response)]
            state["agent_response"] = final_clean_response
            state["conversation_context"] = conversation_context
            state["tool_calls"] = final_tool_calls
            # state["tool_results"] is already cleared if processed by the first block

        except Exception as e:
            print(f"Error in agent node: {str(e)}")
            state["agent_response"] = f"I apologize, but I encountered an error: {str(e)}"
            state["conversation_context"] = conversation_context
            state["tool_calls"] = []
            state["tool_results"] = []

        print("=== End Agent Node Debug ===\n")
        return state
    
    def _generate_response_from_tool_results(self, tool_results: List[Dict], conversation_context: Dict) -> str:
        """Generate a user-friendly response from tool results"""
        for result in tool_results:
            if result.get("action") == "find_next_available": # Explicitly check action
                if result.get("available") and result.get("start_time") and result.get("end_time"):
                    # Check if we have multiple slots
                    available_slots = conversation_context.get("available_slots", [])
                    
                    if len(available_slots) >= 3:
                        # Present multiple slots
                        slot_texts = []
                        for i, slot in enumerate(available_slots[:3], 1):
                            try:
                                start_time = datetime.fromisoformat(slot["start_time"])
                                end_time = datetime.fromisoformat(slot["end_time"])
                                formatted_start = start_time.strftime("%A, %B %d at %I:%M %p")
                                formatted_end = end_time.strftime("%I:%M %p")
                                slot_texts.append(f"Option {i}: {formatted_start} to {formatted_end}")
                            except Exception as e:
                                print(f"Error formatting time: {e}")
                                slot_texts.append(f"Option {i}: {slot['start_time']} to {slot['end_time']}")
                        
                        return f"I found {len(available_slots)} available slots for you:\n" + "\n".join(slot_texts) + "\n\nWhich option would you like me to book?"
                    
                    else:
                        # Present single slot
                        try:
                            start_time = datetime.fromisoformat(result["start_time"])
                            end_time = datetime.fromisoformat(result["end_time"])
                            formatted_start = start_time.strftime("%A, %B %d at %I:%M %p")
                            formatted_end = end_time.strftime("%I:%M %p")
                            
                            return f"Great! I found an available slot for you on {formatted_start} to {formatted_end}. Would you like me to book this meeting for you?"
                        except Exception as e:
                            print(f"Error formatting time: {e}")
                            return f"I found an available slot from {result['start_time']} to {result['end_time']}. Would you like me to book this meeting for you?"
                
                elif result.get("available") is False:
                    # No slots found directly from a tool call result
                    return (
                        f"Sorry, I couldn't find any available slots for your request. "
                        "Would you like to try a different time or duration?"
                    )
            
            elif result.get("action") == "create_meeting": # Explicitly check action
                if result.get("success"):
                    return f"Perfect! I've successfully booked your meeting. {result.get('message', '')} You can view it at: {result.get('event_link', '')}"
                
                elif result.get("error"):
                    return f"I'm sorry, but there was an issue: {result['error']}. Please try again."
            
            elif result.get("action") == "get_events": # Now handle get_events here
                if result.get("success") and result.get("events"):
                    found_events = result["events"]
                    response_message = "Here are the events I found:\n"
                    # Safely access calendar_tool for formatting
                    if self.calendar_tool:
                        for event in found_events:
                            response_message += f"- {event['summary']} from {self.calendar_tool._format_iso_to_natural(event['start'])} to {self.calendar_tool._format_iso_to_natural(event['end'])} ({event['link']})\n"
                    else:
                        for event in found_events:
                            response_message += f"- {event['summary']} from {event['start']} to {event['end']} ({event['link']})\n"
                        response_message += "(Note: Calendar tool is not available for formatting times.)"
                    return response_message
                else:
                    return "I couldn't find any events matching your query. Please try a different name or broader search."
        
        # If we have available slots in conversation context but no specific tool result this turn, present them
        available_slots = conversation_context.get("available_slots", [])
        if available_slots:
            if len(available_slots) >= 3:
                slot_texts = []
                for i, slot in enumerate(available_slots[:3], 1):
                    try:
                        start_time = datetime.fromisoformat(slot["start_time"])
                        end_time = datetime.fromisoformat(slot["end_time"])
                        formatted_start = start_time.strftime("%A, %B %d at %I:%M %p")
                        formatted_end = end_time.strftime("%I:%M %p")
                        slot_texts.append(f"Option {i}: {formatted_start} to {formatted_end}")
                    except Exception as e:
                        print(f"Error formatting time: {e}")
                        slot_texts.append(f"Option {i}: {slot['start_time']} to {slot['end_time']}")
                
                return f"I found {len(available_slots)} available slots for you:\n" + "\n".join(slot_texts) + "\n\nWhich option would you like me to book?"
            else:
                slot = available_slots[0]
                try:
                    start_time = datetime.fromisoformat(slot["start_time"])
                    end_time = datetime.fromisoformat(slot["end_time"])
                    formatted_start = start_time.strftime("%A, %B %d at %I:%M %p")
                    formatted_end = end_time.strftime("%I:%M %p")
                    
                    return f"Great! I found an available slot for you on {formatted_start} to {formatted_end}. Would you like me to book this meeting for you?"
                except Exception as e:
                    print(f"Error formatting time: {e}")
                    return f"I found an available slot from {slot['start_time']} to {slot['end_time']}. Would you like me to book this meeting for you?"
        
        # Default fallback if no specific tool result was processed and no available slots in context
        # This should only be hit if tool_results were empty or unexpected.
        if tool_results:
            # If the only tool result is from find_next_available, and it's successful,
            # we should still present slots and ask for confirmation, not assume booking.
            if len(tool_results) == 1 and tool_results[0].get("action") == "find_next_available" and tool_results[0].get("available"):
                # Let the main logic of _generate_response_from_tool_results handle this
                # which means it will proceed to format and present the available slots.
                pass # Do nothing here, let the fallback below handle presenting slots
            elif tool_results[0].get("action") == "create_meeting" and tool_results[0].get("success"):
                return f"Perfect! I've successfully booked your meeting. {tool_results[0].get('message', '')} You can view it at: {tool_results[0].get('event_link', '')}"
            else:
                return f"I processed the tool results, but couldn't generate a specific response. Tool results: {json.dumps(tool_results[:1], indent=2)}..."
        else:
            return "I've processed your request. Is there anything else I can help you with?"
    
    def _parse_duration(self, text: str) -> Optional[int]:
        """Extracts duration in minutes from text (e.g., '30 min', '1 hour', '2hr')."""
        text_lower = text.lower()

        # Regex for patterns like '30 min', '1 hour', '2.5 hr'
        match = re.search(r'(\d+\.?\d*)\s*(hour|hr|min|minute)s?', text_lower)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit.startswith('hour') or unit.startswith('hr'):
                return int(value * 60)
            elif unit.startswith('min') or unit.startswith('minute'):
                return int(value)
        
        # Check for simple numbers indicating minutes
        if "minute" in text_lower and re.search(r'\d+', text_lower):
            num_match = re.search(r'(\d+)', text_lower)
            if num_match:
                return int(num_match.group(1))
        
        # Default durations based on common phrases
        if "30 min" in text_lower or "half hour" in text_lower:
            return 30
        if "1 hour" in text_lower or "an hour" in text_lower or "60 min" in text_lower:
            return 60
        if "2 hour" in text_lower or "2hr" in text_lower or "120 min" in text_lower:
            return 120
        
        return None

    def _tool_node(self, state: AgentState) -> AgentState:
        """Process tool calls and generate AI messages based on results"""
        print("\n=== Tool Node Debug ===")
        tool_calls = state.get("tool_calls", [])
        conversation_context = state.get("conversation_context", {})
        messages = state.get("messages", [])
        results = []
        print(f"Processing {len(tool_calls)} tool calls")
        if not self.calendar_tool:
            error_msg = "Calendar tool not available"
            print(f"Error: {error_msg}")
            state["agent_response"] = error_msg
            state["messages"] = messages + [Message(role="assistant", content=error_msg)]
            return state
        for tool_call in tool_calls:
            action = tool_call.get("action")
            print(f"Processing tool call: {action}")
            if action and self.calendar_tool:
                try:
                    kwargs = {k: v for k, v in tool_call.items() if k != 'action'}
                    result = self.calendar_tool.invoke(action=action, **kwargs)
                    print(f"Tool result: {result}")
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except json.JSONDecodeError:
                            result = {"error": f"Invalid response: {result}"}
                    results.append(result)
                    # Update conversation context based on tool result
                    if action == "find_next_available":
                        if result.get("available"):
                            # If LLM provided start_time and end_time, use them
                            if "start_time" in tool_call and "end_time" in tool_call:
                                print(f"[DEBUG] Using LLM-parsed start_time: {tool_call['start_time']}, end_time: {tool_call['end_time']}")
                            # Store all available slots in conversation context
                            conversation_context["available_slots"] = result.get("all_slots", [])
                            print(f"[DEBUG] Updated conversation_context with available_slots: {len(conversation_context['available_slots'])} slots")
                        elif result.get("available") is False:
                            conversation_context["available_slots"] = [] # Clear if no slots found
                            print("[DEBUG] No slots found, cleared available_slots in conversation_context")
                    elif action == "create_meeting" and result.get("success"):
                        # After booking, reset context to prevent repeated bookings
                        conversation_context["step"] = "initial"
                    elif action == "get_events" and result.get("success"):
                        # Store event search result in state for agent node to process
                        state["event_search_result"] = result
                        print(f"[DEBUG] Stored event search result: {result}")
                        # Do NOT reset conversation context step here, agent node will handle flow
                except Exception as e:
                    error_message = f"Error executing tool call: {str(e)}"
                    print(f"Tool error: {error_message}")
                    results.append({"error": error_message})
        if results:
            result_message = "Tool results:\n" + "\n".join(
                json.dumps(result, indent=2) for result in results
            )
            messages.append(Message(role="system", content=result_message))
        state["messages"] = messages
        state["conversation_context"] = conversation_context
        state["tool_results"] = results
        print("=== End Tool Node Debug ===\n")
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue to tools or end"""
        tool_calls = state.get("tool_calls", [])
        conversation_context = state.get("conversation_context", {})
        iteration_count = state.get("iteration_count", 0)

        print("\n=== Should Continue Debug ===")
        print(f"  Iteration: {iteration_count}")
        print(f"  Tool Calls: {json.dumps(tool_calls, indent=2)}")
        print(f"  Conversation Context: {json.dumps(conversation_context, indent=2)}")
        print(f"  Has Tool Results: {bool(state.get('tool_results'))}")
        print(f"  Has Event Search Result: {bool(state.get('event_search_result'))}")
        
        # Stop if we've reached the iteration limit
        if iteration_count >= 5:
            print(f"Stopping due to iteration limit ({iteration_count})")
            return "end"
        
        # NEW: If tool results are present, always route back to the agent to process them into a response
        if state.get("tool_results"):
            print("Tool results found, routing back to agent for response generation.")
            return "agent"
        
        # NEW: If an event search result is present, route back to the agent to process it
        if state.get("event_search_result"):
            print("Event search result found, routing back to agent for processing.")
            return "agent"
        
        # Continue if we have tool calls to process
        if tool_calls:
            print(f"Continuing to tools with {len(tool_calls)} tool calls")
            return "continue"
        
        # If we have available slots, we should end and present them to the user
        if conversation_context.get("available_slots", []):
            print("Ending conversation - slots found, presenting to user")
            return "end"
        
        # NEW: If in an information gathering step, end the conversation to wait for user input
        if conversation_context.get("step") in ["gathering_duration", "gathering_time", "need_title", "confirm_booking", "awaiting_event_results_for_details"]:
            print(f"Ending conversation - currently in information gathering step: {conversation_context.get('step')}")
            return "end"
        
        # Check if we're in a state that might need another agent pass
        if conversation_context.get("step") == "find_slots" and not conversation_context.get("available_slots", []):
            if state.get("tool_calls"):
                print("Need to find slots - tool call is pending, continuing to tools")
                return "continue"
            else:
                print("Need to find slots - no tool call, ending to wait for next user input")
                return "end"
        
        print("Ending conversation - no tool calls")
        return "end"
    
    def _parse_context_with_llm(self, user_input: str, current_context: Dict) -> Dict:
        """
        Uses LLM to intelligently parse user input and update conversation context.
        The LLM's output is expected to be a JSON object with specific keys.
        """
        try:
            # We will use a separate, lightweight model or a carefully crafted prompt
            # with the main model for this parsing task.
            # For now, let's use the main model but with a highly structured prompt.

            # Ensure current_context is stringified for the prompt
            context_str = json.dumps(current_context)

            prompt = f"""
            You are an AI assistant tasked with extracting key information and determining the next step
            from a user's input, given the current conversation context.
            Your output MUST be a single-line JSON object. Do NOT include any other text, markdown, or explanations.
            Ensure all string values are properly escaped within the JSON.

            Current Conversation Context: {context_str}
            User Input: "{user_input}"

            Extract the following fields from the user input and current context.
            If a field is not present or cannot be inferred, set its value to `null` (e.g., "field_name": null).

            Fields to extract/infer:
            - "duration": integer (in minutes). E.g., 30, 60, 120.
            - "preferred_time": string (natural language date/time/day/constraint). E.g., "tomorrow at 10 AM", "next Friday", "not on Monday", "after 3 PM".
            - "title": string (proposed meeting title). E.g., "Project Sync", "Follow-up Call".
            - "event_to_follow_up": string (query for a calendar event to schedule after). E.g., "next dim meeting", "stand-up".
            - "selected_slot_index": integer (0-indexed, if user selects a slot, e.g., "option 1" -> 0).
            - "follow_up_offset_days": integer (number of days after the event to schedule, e.g., 1 for "day after", 0 for "same day"), only set if "event_to_follow_up" is also present, or null.
            - "step": string (the next logical step in the conversation flow, based on ALL available info).
                Possible values for "step":
                - "initial": No clear intent, need more info or starting fresh.
                - "gathering_duration": Need duration from user.
                - "gathering_time": Need preferred time from user.
                - "find_slots": Have duration and time, ready to search for slots.
                - "confirm_booking": User is confirming a booking, need to ask for title or book.
                - "need_title": User confirmed booking, but title is missing.
                - "find_event_then_slots": User wants to schedule after a specific event.
                - "awaiting_event_results_for_details": Waiting for event details for general query.

            Priorities for "step" determination:
            1. If `event_to_follow_up` is detected, set `step` to "find_event_then_slots" or "find_event_details" accordingly.
            2. If `user_input` indicates explicit confirmation ("yes", "book it") AND `available_slots` exist in `current_context`, set `step` to "confirm_booking".
            3. If `user_input` provides a `title` AND `current_context.step` was "need_title", set `step` to "confirm_booking" (if slots are available) or "find_slots" (if duration/time are also known).
            4. If both `duration` AND `preferred_time` are available, set `step` to "find_slots".
            5. If only `duration` is available, set `step` to "gathering_time".
            6. If only `preferred_time` is available, set `step` to "gathering_duration".
            7. Otherwise, default to "initial".

            Example 1 (Fresh request):
            User Input: "Book a 30 minute meeting tomorrow morning"
            Output: {{"duration": 30, "preferred_time": "tomorrow morning", "title": null, "event_to_follow_up": null, "selected_slot_index": null, "step": "find_slots", "follow_up_offset_days": null}}

            Example 2 (Follow-up for duration):
            Current Conversation Context: {{"step": "gathering_duration", "preferred_time": "tomorrow at 2 PM", "available_slots": []}}
            User Input: "For 45 minutes"
            Output: {{"duration": 45, "preferred_time": "tomorrow at 2 PM", "title": null, "event_to_follow_up": null, "selected_slot_index": null, "step": "find_slots", "follow_up_offset_days": null}}

            Example 3 (Follow-up for time):
            Current Conversation Context: {{"step": "gathering_time", "duration": 60, "available_slots": []}}
            User Input: "How about next Friday afternoon"
            Output: {{"duration": 60, "preferred_time": "next Friday afternoon", "title": null, "event_to_follow_up": null, "selected_slot_index": null, "step": "find_slots", "follow_up_offset_days": null}}

            Example 4 (Confirm booking):
            Current Conversation Context: {{"step": "confirm_booking", "available_slots": [{{"start_time": "...", "end_time": "..."}}], "duration": 30}}
            User Input: "Yes, book it"
            Output: {{"duration": 30, "preferred_time": null, "title": null, "event_to_follow_up": null, "selected_slot_index": 0, "step": "confirm_booking", "follow_up_offset_days": null}}

            Example 5 (Select specific slot):
            Current Conversation Context: {{"step": "confirm_booking", "available_slots": [{{"start_time": "...", "end_time": "..."}}, {{"start_time": "...", "end_time": "..."}}], "duration": 30}}
            User Input: "Book option 2"
            Output: {{"duration": 30, "preferred_time": null, "title": null, "event_to_follow_up": null, "selected_slot_index": 1, "step": "confirm_booking", "follow_up_offset_days": null}}

            Example 6 (Need title):
            Current Conversation Context: {{"step": "need_title", "available_slots": [{{"start_time": "...", "end_time": "..."}}], "duration": 30}}
            User Input: "Meeting about project review"
            Output: {{"duration": 30, "preferred_time": null, "title": "project review", "event_to_follow_up": null, "selected_slot_index": null, "step": "confirm_booking", "follow_up_offset_days": null}}

            Example 7 (Schedule after event):
            User Input: "Find a 1 hour slot for the day after the team stand-up meeting"
            Output: {{"duration": 60, "preferred_time": null, "title": null, "event_to_follow_up": "team stand-up meeting", "selected_slot_index": null, "step": "find_event_then_slots", "follow_up_offset_days": 1}}

            Example 8 (General event details query):
            User Input: "When is the project review meeting?"
            Output: {{"duration": null, "preferred_time": null, "title": null, "event_to_follow_up": "project review meeting", "selected_slot_index": null, "step": "find_event_details", "follow_up_offset_days": null}}

            Example 9 (Refining time on existing date):
            Current Conversation Context: {{"step": "find_slots", "duration": 60, "preferred_time": "2025-06-24 03:00 PM", "title": null, "available_slots": [], "event_to_follow_up": null, "selected_slot_index": null, "follow_up_offset_days": null}}
            User Input: "how about in the morning?"
            Output: {{"duration": 60, "preferred_time": "2025-06-24 09:00 AM", "title": null, "event_to_follow_up": null, "selected_slot_index": null, "step": "find_slots", "follow_up_offset_days": null}}

            Example 10 (Schedule usual/recurring event):
            User Input: "Schedule our usual sync-up"
            Output: {{"duration": null, "preferred_time": null, "title": "sync-up", "event_to_follow_up": "sync-up", "selected_slot_index": null, "step": "schedule_recurring_event", "follow_up_offset_days": null}}

            Respond with JSON ONLY:
            """.strip() # Using .strip() to remove leading/trailing whitespace/newlines added by the editor.
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            parsed_data = json.loads(self.extract_json_from_response(response.text.strip()))
            print(f"[DEBUG] LLM-parsed context: {json.dumps(parsed_data, indent=2)}")
            return parsed_data
        except Exception as e:
            print(f"Error in LLM context parsing: {e}")
            return {} # Return empty dict on error, to avoid breaking flow

    def extract_json_from_response(self, text):
        # Try to find the first {...} JSON object in the text
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text  # fallback: return the whole text

    def _extract_tool_calls(self, content: str, conversation_context: Dict, user_input: str) -> List[Dict]:
        """Extract tool calls from LLM response, supporting date_range_start/date_range_end and preferred_time/date_str for specific days"""
        import re
        tool_calls = []
        pattern = r'TOOL_CALL:\s*(\{[^}]+\})'
        matches = re.findall(pattern, content)
        for match in matches:
            try:
                tool_call = json.loads(match)
                if "duration_minutes" in tool_call and isinstance(tool_call["duration_minutes"], str):
                    try:
                        tool_call["duration_minutes"] = int(tool_call["duration_minutes"])
                    except ValueError:
                        print(f"Warning: Could not convert duration_minutes '{tool_call['duration_minutes']}' to int")
                        continue
                # Map start_date/end_date to date_range_start/date_range_end for backward compatibility
                if tool_call.get("start_date") and not tool_call.get("date_range_start"):
                    tool_call["date_range_start"] = tool_call["start_date"]
                if tool_call.get("end_date") and not tool_call.get("date_range_end"):
                    tool_call["date_range_end"] = tool_call["end_date"]
                
                if tool_call.get("action") == "find_next_available":
                    # CRITICAL: ALWAYS use preferred_time and duration from conversation_context if available,
                    # overriding any potentially incorrect values from the LLM's raw TOOL_CALL output.
                    if conversation_context.get("preferred_time"):
                        tool_call["date_str"] = conversation_context["preferred_time"]
                        print(f"[DEBUG] [FORCE OVERRIDE] Using conversation_context['preferred_time'] for date_str: '{conversation_context['preferred_time']}'")
                    
                    if conversation_context.get("duration"):
                        tool_call["duration_minutes"] = conversation_context["duration"]
                        print(f"[DEBUG] [FORCE OVERRIDE] Using conversation_context['duration'] for duration_minutes: '{conversation_context['duration']}'")
                tool_calls.append(tool_call)
            except json.JSONDecodeError as e:
                print(f"Error parsing tool call JSON: {e}")
                continue
        return tool_calls
    
    def _clean_response_content(self, content: str) -> str:
        """Clean response content by removing tool call markers and code blocks"""
        # Remove TOOL_CALL patterns
        cleaned = re.sub(r'TOOL_CALL:\s*\{[^}]+\}', '', content)
        
        # Remove code block markers
        cleaned = re.sub(r'```[^`]*```', '', cleaned)
        cleaned = re.sub(r'```tool_code\s*```', '', cleaned)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def process_message(self, user_input: str, conversation_history: List[Dict] = None, selected_slot_index: Optional[int] = None) -> Dict[str, Any]:
        """Process a user message and return the agent's response"""
        print(f"\n=== Processing Message: {user_input} ===")
        
        # Initialize state
        initial_state = AgentState({
            "messages": [],
            "user_input": user_input,
            "agent_response": "",
            "tool_calls": [],
            "conversation_context": {},
            "iteration_count": 0,
            "tool_results": [],
            "event_search_result": None
        })
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                message = Message(
                    role=msg["role"],
                    content=msg["content"]
                )
                initial_state["messages"].append(message)
                
                # Restore conversation context from last assistant message
                if msg["role"] == "assistant" and "conversation_context" in msg:
                    initial_state["conversation_context"] = msg["conversation_context"]
        
        # If a specific slot was selected, add it to the conversation context for the agent to use
        if selected_slot_index is not None:
            initial_state["conversation_context"]["selected_slot_index"] = selected_slot_index
            print(f"[DEBUG] process_message: Added selected_slot_index {selected_slot_index} to initial_state context.")
        
        try:
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            response = result["agent_response"]
            conversation_context = result.get("conversation_context", {})
            
            # Check if we have available slots
            available_slots = conversation_context.get("available_slots", [])
            has_slots = len(available_slots) > 0
            
            # Create response data
            response_data = {
                "response": response,
                "success": True,
                "tool_calls_made": len(result.get("tool_calls", [])) > 0,
                "conversation_context": conversation_context,
                "has_slots": has_slots,
                "available_slots": available_slots,
                "debug_info": {
                    "current_step": conversation_context.get("step"),
                    "last_action": conversation_context.get("last_action"),
                    "parsed_input": conversation_context.get("parsed_input"),
                    "tool_calls": result.get("tool_calls", [])
                }
            }
            
            # Update conversation history
            updated_history = conversation_history or []
            
            # Add user message to history
            updated_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Add assistant response to history
            updated_history.append({
                "role": "assistant",
                "content": response,
                "conversation_context": conversation_context
            })
            
            response_data["conversation_history"] = updated_history
            
            return response_data
        
        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                "success": False,
                "error": str(e),
                "conversation_context": initial_state.get("conversation_context", {}),
                "has_slots": False,
                "available_slots": [],
                "conversation_history": conversation_history or [],
                "debug_info": {
                    "error": str(e),
                    "current_step": "error"
                }
            }

# Singleton instance
_agent_instance = None

def get_agent() -> SmartSchedulerAgent:
    """Get or create the agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SmartSchedulerAgent()
    return _agent_instance