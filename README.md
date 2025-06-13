# Smart Scheduler AI Agent

A smart meeting scheduling assistant that uses AI to understand natural language requests and manage your calendar. The application features voice input, text-to-speech responses, and intelligent calendar management.
Check out the video here.
https://www.loom.com/share/5dca2e517e744bdca5b7a43f540efc68?sid=5394df23-ab1d-48db-9aee-ee20f9b8e3d2

## Features

- 🤖 AI-powered natural language processing for scheduling requests
- 🎤 Voice input support with real-time speech recognition
- 🔊 Text-to-speech responses using Eleven Labs
- 📅 Google Calendar integration for managing meetings
- 💬 Interactive chat interface with real-time updates
- 📱 Responsive design for both desktop and mobile
- 🔄 Automatic timezone handling (IST/UTC)

## Prerequisites

- Python 3.8 or higher
- Google Cloud Platform account with Calendar API enabled
- Gemini API key
- Eleven Labs API key
- Modern web browser with microphone support

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd smart-scheduler
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env`
   - Fill in your API keys and credentials:
     - Get Google Calendar API credentials from [Google Cloud Console](https://console.cloud.google.com)
     - Get Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
     - Get Eleven Labs API key from [Eleven Labs](https://elevenlabs.io)

5. **Configure Google Calendar API**
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select an existing one
   - Enable the Google Calendar API
   - Create OAuth 2.0 credentials
   - Add `http://localhost:5000/oauth2callback` as an authorized redirect URI
   - Download the credentials and update your `.env` file

## Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the application**
   - Open your browser and go to `http://localhost:5000`
   - On first run, you'll be prompted to authenticate with Google Calendar
   - Grant the necessary permissions when prompted

## Usage

1. **Scheduling a Meeting**
   - Type or speak your scheduling request (e.g., "Schedule a 1-hour meeting tomorrow afternoon")
   - The AI will parse your request and check calendar availability
   - Select from available time slots
   - Confirm the meeting details

2. **Voice Commands**
   - Click and hold the microphone button to start recording
   - Speak your request clearly
   - Release the button to send your request

3. **Quick Actions**
   - Use the quick action buttons for common scheduling requests
   - Customize the quick actions in the interface

## Troubleshooting

1. **Google Calendar Authentication Issues**
   - Ensure your OAuth credentials are correctly configured
   - Check that the redirect URI matches exactly
   - Clear browser cookies if authentication fails

2. **Voice Recognition Problems**
   - Check microphone permissions in your browser
   - Ensure you're in a quiet environment
   - Speak clearly and at a normal pace

3. **API Key Issues**
   - Verify all API keys are correctly set in `.env`
   - Check API key quotas and limits
   - Ensure you have enabled all necessary APIs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Calendar API
- Google Gemini AI
- Eleven Labs
- Flask Framework
- SpeechRecognition library 
