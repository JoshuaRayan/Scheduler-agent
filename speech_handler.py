import pyaudio
import speech_recognition as sr
import numpy as np
import threading
import queue
import time

class SpeechHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.callback = None
        
        # Optimize recognition settings
        self.recognizer.energy_threshold = 300  # Lower threshold for better sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.5  # Shorter pause threshold for faster response
        
    def start_recording(self, callback):
        """Start recording audio with optimized settings"""
        self.callback = callback
        self.is_recording = True
        
        # Configure audio stream with optimized settings
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,  # Lower sample rate for faster processing
            input=True,
            frames_per_buffer=1024,  # Smaller buffer size for lower latency
            stream_callback=self._audio_callback
        )
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.start()
        
    def stop_recording(self):
        """Stop recording and clean up resources"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.processing_thread:
            self.processing_thread.join()
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
        
    def _process_audio(self):
        """Process audio data in a separate thread"""
        audio_data = bytearray()
        
        while self.is_recording:
            try:
                # Get audio data from queue with timeout
                data = self.audio_queue.get(timeout=0.1)
                audio_data.extend(data)
                
                # Process in chunks to reduce latency
                if len(audio_data) >= 16000:  # Process every 1 second of audio
                    audio_data_np = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Convert to AudioData for recognition
                    audio_data_sr = sr.AudioData(
                        audio_data_np.tobytes(),
                        sample_rate=16000,
                        sample_width=2
                    )
                    
                    try:
                        # Use Google's speech recognition with optimized settings
                        text = self.recognizer.recognize_google(
                            audio_data_sr,
                            language='en-US',
                            show_all=False
                        )
                        
                        if text and self.callback:
                            self.callback(text)
                            
                    except sr.UnknownValueError:
                        pass  # Ignore unrecognized speech
                    except sr.RequestError as e:
                        print(f"Recognition error: {e}")
                        
                    # Clear processed audio data
                    audio_data = bytearray()
                    
            except queue.Empty:
                continue
                
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_recording()
        self.audio.terminate() 