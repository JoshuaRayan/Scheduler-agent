import os
import io
import queue
import re
import time
import json

from dotenv import load_dotenv

import pyaudio
import pygame
import speech_recognition as sr # For simple synchronous STT
import httpx # For simple synchronous TTS

import google.generativeai as genai

# Assuming llm_provider.py exists and provides get_llm
from llm_provider import get_llm
import config # To access DEEPGRAM_API_KEY
import tools # Import the tools module for function calling

load_dotenv()

# --- API Keys and Clients ---
DEEPGRAM_API_KEY = config.DEEPGRAM_API_KEY # Assuming DEEPGRAM_API_KEY is in config.py or .env

# --- Audio Recording Parameters ---
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

# Initialize audio playback (Pygame for playing TTS audio)
# Pygame mixer needs to be initialized only once
pygame.mixer.init()

# --- LLM Interaction Function ---
def get_gemini_response(chat: genai.GenerativeModel.start_chat, prompt: str) -> str:
    """
    Gets a response from the Gemini LLM for a given prompt using a persistent chat session.
    Handles potential function calls from the LLM.
    
    Args:
        chat: The persistent Gemini chat object.
        prompt: The text prompt to send to the LLM.
        
    Returns:
        The LLM's text response, or a simplified error message.
    """
    try:
        response = chat.send_message(prompt,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 2048
            }
        )
        
        # Handle function calls if the LLM makes one
        while True:
            function_calls = []
            for part in response.parts:
                if part.function_call:
                    function_calls.append(part.function_call)
            
            if function_calls:
                function_responses = []
                for fc in function_calls:
                    tool_function = getattr(tools, fc.name)
                    args = {key: value for key, value in fc.args.items()}
                    
                    print(f"[DEBUG] Calling tool: {fc.name} with args: {args}")
                    tool_output = tool_function(**args)
                    print(f"[DEBUG] Tool output: {tool_output}")
                    
                    function_responses.append(
                        genai.protos.Part(function_response=genai.protos.FunctionResponse(
                            name=fc.name,
                            response=tool_output
                        ))
                    )
                
                # Send all function responses back to the LLM in one go
                response = chat.send_message(function_responses)

            elif response.text:
                return response.text.strip()
            else:
                return "I'm sorry, I couldn't generate a text response after tool execution."

    except Exception as e:
        error_message = str(e)
        print(f"An error occurred with the LLM or tool execution: {error_message}")
        if "quota" in error_message.lower() or "429" in error_message:
            return "My apologies, I've hit my usage limit. Please try again in a moment."
        else:
            return "I encountered an unexpected error with the AI. Please try again."

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

    print(f"AI: {text}") # Print the text being spoken *before* the API call
    url = "https://api.deepgram.com/v1/speak"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg", # Explicitly request MP3 audio for pygame
    }
    payload = {
        "text": text,
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
            
            # 2. Get response from Gemini LLM (now handles tool calls internally)
            print("AI Thinking...")
            final_ai_response = get_gemini_response(chat, user_input) 
            
            # 3. Speak the AI's final response (TTS) synchronously
            speak_text_sync(final_ai_response)
            
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