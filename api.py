from flask import Flask, request, jsonify, render_template
import datetime
import pytz
import requests
import hmac
import hashlib
from flask_socketio import SocketIO, emit
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
import urllib.parse
from flask_cors import CORS # Import CORS

from google_calendar import find_free_slots, book_meeting, get_calendar_service_instance, update_appointment, delete_appointment
from config import BLAND_AI_API_KEY, BLAND_AI_WEBHOOK_SECRET, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, BLAND_AI_INBOUND_NUMBER

app = Flask(__name__)

# Define origins for CORS
# In production, replace with your actual Next.js frontend URL
origins = [
    "http://localhost:3000", # Your Next.js development server
    "http://localhost:8000", # If the frontend is ever served from the same origin as Flask itself (less common)\
    "https://1e7a-49-206-134-217.ngrok-free.app",
    # Add your ngrok URL if you're accessing backend from Next.js via ngrok, e.g., "https://your-unique-id.ngrok-free.app"
]

socketio = SocketIO(app, cors_allowed_origins=origins) # Use defined origins for SocketIO

# Apply CORS middleware to Flask app.
# The resources parameter is important for specific route-level CORS.
CORS(app, resources={r"/*": {"origins": origins}}) # Initialize CORS for your Flask app

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
IST = pytz.timezone('Asia/Kolkata')
active_calls = {}

def get_base_url():
    if request.headers.get('X-Forwarded-Proto'):
        return f"{request.headers['X-Forwarded-Proto']}://{request.headers['Host']}"
    return request.url_root.rstrip('/')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inbound-calls')
def inbound_calls():
    return render_template('inbound_calls.html')

@app.route('/twilio/message_and_hangup', methods=['POST'])
def twilio_message_and_hangup():
    message = request.args.get('message', 'we will issue a call back to your number soon.')
    response_twiml = VoiceResponse()
    response_twiml.say(message)
    response_twiml.hangup()
    return str(response_twiml), 200, {'Content-Type': 'text/xml'}

@app.route('/calendar/v3/freeBusy', methods=['POST'])
def get_free_busy_slots():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    time_min_str = data.get('timeMin')
    time_max_str = data.get('timeMax')
    duration_minutes = data.get('meeting_duration', 30) # Default to 30 minutes if not provided
    time_zone_str = data.get('timeZone', 'Asia/Kolkata') # Default to Asia/Kolkata if not provided

    # Get the timezone object
    try:
        requested_timezone = pytz.timezone(time_zone_str)
    except pytz.UnknownTimeZoneError:
        return jsonify({"error": f"Invalid timeZone: {time_zone_str}"}), 400

    if not time_min_str or not time_max_str:
        return jsonify({"error": "timeMin and timeMax are required"}), 400

    if not isinstance(duration_minutes, int) or duration_minutes <= 0:
        return jsonify({"error": "meeting_duration must be a positive integer"}), 400
    # parsing logic to follow
    try:
        if time_min_str.endswith('Z'):
            naive_start_dt = datetime.datetime.fromisoformat(time_min_str.replace('Z', ''))
            start_dt_localized = requested_timezone.localize(naive_start_dt)
        else:
            start_dt_localized = datetime.datetime.fromisoformat(time_min_str)
            if start_dt_localized.tzinfo is None:
                start_dt_localized = requested_timezone.localize(start_dt_localized)
            else:
                start_dt_localized = start_dt_localized.astimezone(requested_timezone)

        if time_max_str.endswith('Z'):
            naive_end_dt = datetime.datetime.fromisoformat(time_max_str.replace('Z', ''))
            end_dt_localized = requested_timezone.localize(naive_end_dt)
        else:
            end_dt_localized = datetime.datetime.fromisoformat(time_max_str)
            if end_dt_localized.tzinfo is None:
                end_dt_localized = requested_timezone.localize(end_dt_localized)
            else:
                end_dt_localized = end_dt_localized.astimezone(requested_timezone)

        
        free_slots = find_free_slots(start_dt_localized, end_dt_localized, duration_minutes=duration_minutes)

        top_3_free_slots = free_slots[:3]

        formatted_slots = [
            {"start": slot_start.isoformat(), "end": slot_end.isoformat(), "timeZone": time_zone_str}
            for slot_start, slot_end in top_3_free_slots
        ]
        return jsonify({"free_slots": formatted_slots}), 200

    except ValueError as e:
        return jsonify({"error": f"Invalid date/time format: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

@app.route('/calendar/v3/events', methods=['POST'])
def book_new_meeting():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    start_time_str = data.get('start')
    end_time_str = data.get('end')
    summary = data.get('summary', 'Appointment') 
    phone_number = data.get('phone_number') 
    time_zone_str = data.get('timeZone', 'Asia/Kolkata')

    
    try:
        requested_timezone = pytz.timezone(time_zone_str)
    except pytz.UnknownTimeZoneError:
        return jsonify({"error": f"Invalid timeZone: {time_zone_str}"}), 400

    if not start_time_str or not end_time_str:
        return jsonify({"error": "'start' and 'end' are required"}), 400

    try:
       
        if start_time_str.endswith('Z'):
            naive_start_dt = datetime.datetime.fromisoformat(start_time_str.replace('Z', ''))
            start_dt_localized = requested_timezone.localize(naive_start_dt)
        else:
            start_dt_localized = datetime.datetime.fromisoformat(start_time_str)
            if start_dt_localized.tzinfo is None:
                start_dt_localized = requested_timezone.localize(start_dt_localized)
            else:
                start_dt_localized = start_dt_localized.astimezone(requested_timezone)

        if end_time_str.endswith('Z'):
            naive_end_dt = datetime.datetime.fromisoformat(end_time_str.replace('Z', ''))
            end_dt_localized = requested_timezone.localize(naive_end_dt)
        else:
            end_dt_localized = datetime.datetime.fromisoformat(end_time_str)
            if end_dt_localized.tzinfo is None:
                end_dt_localized = requested_timezone.localize(end_dt_localized)
            else:
                end_dt_localized = end_dt_localized.astimezone(requested_timezone)

        
        event_link = book_meeting(start_dt_localized, end_dt_localized, summary, phone_number)
        return jsonify({"message": "Meeting booked successfully", "event_link": event_link}), 200

    except ValueError as e:
        return jsonify({"error": f"Invalid date/time format: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

@app.route('/calendar/v3/appointments/update', methods=['POST'])
def update_existing_appointment():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    phone_number = data.get('phone_number')
    old_summary = data.get('old_summary')
    new_start_time_str = data.get('new_start') 
    new_summary = data.get('new_summary') 
    time_zone_str = data.get('timeZone', 'Asia/Kolkata') 

    if not phone_number or not old_summary:
        return jsonify({"error": "Phone number and old_summary are required."}), 400

    try:
        requested_timezone = pytz.timezone(time_zone_str)
    except pytz.UnknownTimeZoneError:
        return jsonify({"error": f"Invalid timeZone: {time_zone_str}"}), 400

    service_instance = get_calendar_service_instance()
    matching_events = service_instance.get_events_by_phone_number(phone_number)

    if not matching_events:
        return jsonify({"message": "No appointments found for the given phone number."}), 200

    
    lower_old_summary = old_summary.lower()
    filtered_events = [
        event for event in matching_events
        if lower_old_summary in event.get('summary', '').lower()
    ]

    if not filtered_events:
        return jsonify({"message": "No appointments found with that phone number and matching title."}), 200

    updated_count = 0
    for event in filtered_events:
        event_id = event['id']
        
        # Get original start and end times
        original_start_str = event['start'].get('dateTime')
        original_end_str = event['end'].get('dateTime')

        if not original_start_str or not original_end_str:
            print(f"Skipping event {event_id}: Missing start or end time.")
            continue

        original_start_dt_utc = datetime.datetime.fromisoformat(original_start_str.replace('Z', '+00:00'))
        original_end_dt_utc = datetime.datetime.fromisoformat(original_end_str.replace('Z', '+00:00'))
        original_duration = original_end_dt_utc - original_start_dt_utc
        current_start_dt = original_start_dt_utc.astimezone(requested_timezone)
        current_end_dt = original_end_dt_utc.astimezone(requested_timezone)
        new_effective_start_dt = current_start_dt
        
        if new_start_time_str:
            # Parse new_start_time_str, handling 'Z' suffix and localizing
            if new_start_time_str.endswith('Z'):
                naive_new_start_dt = datetime.datetime.fromisoformat(new_start_time_str.replace('Z', ''))
                new_effective_start_dt = requested_timezone.localize(naive_new_start_dt)
            else:
                new_effective_start_dt = datetime.datetime.fromisoformat(new_start_time_str)
                if new_effective_start_dt.tzinfo is None:
                    new_effective_start_dt = requested_timezone.localize(new_effective_start_dt)
                else:
                    new_effective_start_dt = new_effective_start_dt.astimezone(requested_timezone)
        
        new_effective_end_dt = new_effective_start_dt + original_duration
        effective_new_summary = new_summary if new_summary is not None else event.get('summary', 'Appointment')

    
        update_result = update_appointment(
            event_id,
            new_effective_start_dt,
            new_effective_end_dt,
            effective_new_summary,
            phone_number # Pass phone number to ensure it's kept in description
        )
        if "successfully" in update_result.lower():
            updated_count += 1
        else:
            print(f"Failed to update event {event_id}: {update_result}")

    if updated_count > 0:
        return jsonify({"message": f"Successfully updated {updated_count} appointment(s)."}), 200
    else:
        return jsonify({"message": "No matching appointments were updated."}), 200

@app.route('/calendar/v3/appointments/delete', methods=['POST'])
def delete_existing_appointment():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    phone_number = data.get('phone_number')
    summary = data.get('summary') # Get summary from payload

    if not phone_number:
        return jsonify({"error": "Phone number is required for verification."}), 400

    try:
        message = delete_appointment(phone_number, summary) # Pass summary to the function
        return jsonify({"message": message}), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

@app.route('/bland-ai/call', methods=['POST'])
def make_bland_ai_call():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    phone_number = data.get('phone_number')
    if not phone_number:
        return jsonify({"error": "Phone number is required."}), 400

    headers = {
        'Authorization': BLAND_AI_API_KEY,
        'Content-Type': 'application/json'
    }

    call_data = {
        "phone_number": phone_number,
        "voice": "June",
        "wait_for_greeting": False,
        "record": True,
        "answered_by_enabled": True,
        "noise_cancellation": False,
        "interruption_threshold": 100,
        "block_interruptions": False,
        "max_duration": 12,
        "model": "base",
        "language": "en",
        "background_track": "none",
        "endpoint": "https://api.bland.ai",
        "voicemail_action": "hangup",
        "pathway_id": "c1496235-5736-44c5-b940-f10e37fd0d5b",
        "pathway_version": 2
    }

    try:
        response = requests.post('https://api.bland.ai/v1/calls', json=call_data, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        return jsonify(response.json()), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to make Bland AI call: {e}"}), 500

@app.route('/bland-ai/redirect_and_end_call', methods=['POST'])
def redirect_and_end_call():
    data = request.json
    bland_ai_call_id = data.get('bland_ai_call_id')
    message_to_speak = data.get('message', 'Thank you for calling. Goodbye!')

    if not bland_ai_call_id:
        return jsonify({"error": "Bland AI Call ID is required."}), 400

    call_info = active_calls.get(bland_ai_call_id)

    if not call_info:
        return jsonify({"error": "No active call found for the given Bland AI Call ID."}), 404

    twilio_call_sid = call_info.get('twilio_sid')
    from_number = call_info.get('from_number')

    if not twilio_call_sid:
        return jsonify({"error": "Twilio CallSid not found for this Bland AI Call ID. Cannot redirect or end Twilio leg."}), 400

    bland_ai_stop_successful = False
    bland_ai_error_message = ""

    try:
        # Step 1: Attempt to stop the Bland AI call
        print(f"Attempting to stop Bland AI call {bland_ai_call_id}.")
        stop_bland_headers = {
            'Authorization': BLAND_AI_API_KEY,
            'Content-Type': 'application/json'
        }
        stop_bland_response = requests.post(f'https://api.bland.ai/v1/calls/{bland_ai_call_id}/stop', headers=stop_bland_headers)
        stop_bland_response.raise_for_status()
        print(f"Bland AI call {bland_ai_call_id} stopped successfully.")
        bland_ai_stop_successful = True
    except requests.exceptions.RequestException as e:
        bland_ai_error_message = f"Failed to stop Bland AI call: {e}"
        if e.response is not None:
            try:
                error_json = e.response.json()
                bland_ai_error_message += f" - Bland AI response: {error_json}"
            except ValueError:
                bland_ai_error_message += f" - Bland AI raw response: {e.response.text}"
        print(bland_ai_error_message)

    try:
        print(f"Attempting to redirect Twilio CallSid {twilio_call_sid} to play message and hang up.")
        redirect_url = f"{get_base_url()}/twilio/message_and_hangup?message={urllib.parse.quote(message_to_speak)}"
        twilio_client.calls(twilio_call_sid).update(method='POST', url=redirect_url)
        print(f"Twilio CallSid {twilio_call_sid} redirected to {redirect_url}")

        # Clean up the active_calls entry
        if bland_ai_call_id in active_calls:
            del active_calls[bland_ai_call_id]

        status_message = "Call redirected and termination attempted."

        return jsonify({"status": "success", "message": status_message}), 200

    except Exception as e:
        error_message = f"Error redirecting Twilio call: {e}"
        print(error_message)
        return jsonify({"status": "error", "message": error_message}), 500

@app.route('/bland-ai/list_calls', methods=['GET'])
def list_bland_ai_calls():
    headers = {
        'Authorization': BLAND_AI_API_KEY,
    }
    try:
        response = requests.get('https://api.bland.ai/v1/calls', headers=headers)
        response.raise_for_status()
        calls_data = response.json().get('calls', [])
        
        active_inbound_calls = []
        for call in calls_data:
            call_id = call.get('call_id')
            
            current_twilio_sid = active_calls.get(call_id, {}).get('twilio_sid')
            if current_twilio_sid is None:
                current_twilio_sid = call.get('sid')
            if current_twilio_sid is None and call.get('inbound') == True and call_id:
                detail_response = requests.get(f'https://api.bland.ai/v1/calls/{call_id}', headers=headers)
                detail_response.raise_for_status()
                detailed_call_data = detail_response.json()
                current_twilio_sid = detailed_call_data.get('sid')
                from_number = detailed_call_data.get('from')
                if current_twilio_sid and from_number:
                    active_calls[call_id] = {
                        'twilio_sid': current_twilio_sid,
                        'from_number': from_number
                    }
            
            if call.get('inbound') == True and call.get('queue_status') in ['started', 'allocated', 'queued', 'new', 'pending']:
                active_inbound_calls.append({
                    'call_id': call.get('call_id'),
                    'from_number': call.get('from'),
                    'to_number': call.get('to'),
                    'status': call.get('queue_status'),
                    'created_at': call.get('created_at'),
                    'twilio_call_sid': current_twilio_sid # Use the retrieved Twilio CallSid
                })
            elif call.get('inbound') == True and call.get('queue_status') == 'completed' and \
                 (datetime.datetime.now(pytz.utc) - datetime.datetime.fromisoformat(call.get('created_at').replace('Z', '+00:00'))).total_seconds() < 300: # Last 5 minutes
                active_inbound_calls.append({
                    'call_id': call.get('call_id'),
                    'from_number': call.get('from'),
                    'to_number': call.get('to'),
                    'status': call.get('queue_status'),
                    'created_at': call.get('created_at'),
                    'twilio_call_sid': current_twilio_sid # Use the retrieved Twilio CallSid
                })

        return jsonify({"active_inbound_calls": active_inbound_calls}), 200

    except requests.exceptions.RequestException as e:
        error_message = f"Failed to retrieve calls from Bland AI: {e}"
        if e.response is not None:
            try:
                error_json = e.response.json()
                error_message += f" - Bland AI response: {error_json}"
            except ValueError:
                error_message += f" - Bland AI raw response: {e.response.text}"
        print(error_message)
        return jsonify({"error": error_message}), e.response.status_code if e.response is not None else 500

@app.route('/bland-ai/webhook', methods=['POST']) # webhook for call events
def bland_ai_webhook():
    signature = request.headers.get('X-Bland-Signature')
    if not signature:
        return jsonify({"error": "X-Bland-Signature header missing"}), 401

    raw_data = request.get_data()

    expected_signature = hmac.new(
        BLAND_AI_WEBHOOK_SECRET.encode('utf-8'),
        raw_data,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, signature):
        return jsonify({"error": "Invalid signature"}), 401
    
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400
    
    call_id = data.get('call_id')
    from_number = data.get('from')
    twilio_call_sid = data.get('sid')

    print(f"Received Bland AI webhook data for call_id {call_id}: {data}")

    if call_id:
        active_calls[call_id] = {
            'twilio_sid': twilio_call_sid,
            'from_number': from_number
        }

    emit('transcript', data, broadcast=True, namespace='/') # Emit transcript data via WebSocket

    # Bland AI expects a 200 OK response
    return jsonify({"status": "success"}), 200

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@app.route('/bland-ai/transcript/<call_id>', methods=['GET']) # this is used to get the live transcript of a call
def get_bland_ai_transcript(call_id):
    headers = {
        'Authorization': BLAND_AI_API_KEY,
    }
    try:
        response = requests.get(f'https://api.bland.ai/v1/calls/{call_id}', headers=headers)
        response.raise_for_status()
        return jsonify(response.json()), 200
    except requests.exceptions.RequestException as e:
        error_message = f"Failed to retrieve transcript from Bland AI: {e}"
        if e.response is not None:
            try:
                error_json = e.response.json()
                error_message += f" - Bland AI response: {error_json}"
            except ValueError:
                error_message += f" - Bland AI raw response: {e.response.text}"
        print(error_message)
        return jsonify({"error": error_message}), e.response.status_code if e.response is not None else 500

if __name__ == '__main__':
    socketio.run(app, debug=True, port=8000)