from flask import Flask, render_template_string, request, Response
from flask_socketio import SocketIO, emit
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, VitsModel, AutoTokenizer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from deep_translator import GoogleTranslator
import io
import numpy as np
import eventlet
import requests
import json

# --- Backend Application ---

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# --- Load the STT and TTS models ---

# STT: Gujarati Speech to Gujarati Text
try:
    stt_processor = Wav2Vec2Processor.from_pretrained("addy88/wav2vec2-gujarati-stt")
    stt_model = Wav2Vec2ForCTC.from_pretrained("addy88/wav2vec2-gujarati-stt")
    print("STT Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading STT model: {e}")
    stt_processor = None
    stt_model = None

# TTS: Gujarati Text to Gujarati Speech
try:
    tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-guj")
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-guj")
    print("TTS Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading TTS model: {e}")
    tts_tokenizer = None
    tts_model = None

# New: Initialize DeepTranslator models
try:
    # GoogleTranslator uses simple language codes: 'gu' for Gujarati and 'en' for English
    guj_en_translator = GoogleTranslator(source='gu', target='en')
    en_guj_translator = GoogleTranslator(source='en', target='gu')
    print("DeepTranslator models loaded successfully.")
except Exception as e:
    print(f"Error initializing DeepTranslator: {e}")
    guj_en_translator = None
    en_guj_translator = None

# A dictionary to store audio buffers for each session (client)
session_data = {}
TARGET_SAMPLE_RATE = 16000 # The STT model's required sample rate

# URL of your FastAPI backend
FASTAPI_URL = "http://127.0.0.1:8000/api/query"

def get_llm_response(query: str) -> str:
    """Sends a query to the FastAPI backend and returns the English response."""
    try:
        payload = {"query": query, "role": "student"}
        headers = {"Content-Type": "application/json"}
        response = requests.post(FASTAPI_URL, data=json.dumps(payload), headers=headers, timeout=60)
        response.raise_for_status()
        return response.json().get("answer", "I'm sorry, I couldn't get a response from the AI assistant.")
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with FastAPI backend: {e}")
        return "I'm sorry, I am unable to connect to my knowledge base right now. Please try again."

@app.route('/')
def index():
    """Renders the single-page HTML application."""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gujarati Speech App</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body {
            font-family: 'Inter', sans-serif;
            background: #f3f4f6;
        }
    </style>
</head>
<body class="flex items-center justify-center min-h-screen p-4">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-2xl">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">
            Gujarati Speech App
        </h1>
        <p class="text-center text-gray-500 mb-8">
            This app demonstrates both Speech-to-Text and Text-to-Speech in real time.
        </p>

        <div class="flex flex-col items-center justify-center space-y-6">
            <h2 class="text-xl font-semibold text-gray-800">Speech-to-Text</h2>
            <div class="flex space-x-4">
                <button id="recordButton" class="bg-indigo-600 text-white font-semibold py-3 px-6 rounded-full shadow-lg hover:bg-indigo-700 transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-opacity-50">
                    Start Recording
                </button>
                <button id="cancelButton" class="bg-gray-600 text-white font-semibold py-3 px-6 rounded-full shadow-lg hover:bg-gray-700 transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50 hidden">
                    Cancel Recording
                </button>
            </div>
            <div id="status" class="text-lg font-medium text-gray-700">Ready</div>
        </div>

        <div class="mt-8 bg-gray-100 p-6 rounded-xl border border-gray-200">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">Original:</h2>
            <div id="transcriptionOutput" class="text-gray-900 text-base min-h-[50px] overflow-auto">
                <p>Transcription will appear here...</p>
            </div>
            <h3 class="text-lg font-medium text-gray-700 mt-4">English Translation:</h3>
            <div id="englishTranslationOutput" class="text-gray-900 text-base min-h-[50px] overflow-auto">
                <p>English translation will appear here...</p>
            </div>
            <h3 class="text-lg font-medium text-gray-700 mt-4">Gujarati Round-trip Translation:</h3>
            <div id="translationOutput" class="text-gray-900 text-base min-h-[50px] overflow-auto">
                <p>Final translation will appear here...</p>
            </div>
        </div>

        <div class="mt-8 flex flex-col items-center justify-center space-y-6">
            <h2 class="text-xl font-semibold text-gray-800">Text-to-Speech</h2>
            <div class="w-full">
                <textarea id="ttsInput" class="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="Type Gujarati text to convert to speech..."></textarea>
            </div>
            <button id="ttsButton" class="bg-green-600 text-white font-semibold py-3 px-6 rounded-full shadow-lg hover:bg-green-700 transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50">
                Synthesize Speech
            </button>
            <div id="ttsStatus" class="text-lg font-medium text-gray-700 hidden">Synthesizing...</div>
        </div>

        <div id="errorBox" class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative hidden mt-4" role="alert">
            <strong class="font-bold">Error!</strong>
            <span class="block sm:inline" id="errorMessage"></span>
        </div>
    </div>

    <script>
        const recordButton = document.getElementById('recordButton');
        const cancelButton = document.getElementById('cancelButton');
        const ttsButton = document.getElementById('ttsButton');
        const ttsInput = document.getElementById('ttsInput');
        const statusElement = document.getElementById('status');
        const ttsStatusElement = document.getElementById('ttsStatus');
        const transcriptionOutput = document.getElementById('transcriptionOutput');
        const englishTranslationOutput = document.getElementById('englishTranslationOutput');
        const translationOutput = document.getElementById('translationOutput');
        const errorBox = document.getElementById('errorBox');
        const errorMessage = document.getElementById('errorMessage');

        let socket;
        let audioContext;
        let scriptProcessor;
        let isRecording = false;
        let ttsAudioQueue = [];
        let isPlaying = false;
        let audioPlayer = new Audio();

        function displayError(message) {
            errorMessage.textContent = message;
            errorBox.classList.remove('hidden');
            statusElement.textContent = 'Error';
        }

        function resetOutputs() {
            transcriptionOutput.innerHTML = '<p>Transcription will appear here...</p>';
            englishTranslationOutput.innerHTML = '<p>English translation will appear here...</p>';
            translationOutput.innerHTML = '<p>Final translation will appear here...</p>';
        }

        function playNextAudio() {
            if (ttsAudioQueue.length > 0 && !isPlaying) {
                isPlaying = true;
                const audioBlob = ttsAudioQueue.shift();
                const audioUrl = URL.createObjectURL(audioBlob);
                audioPlayer.src = audioUrl;
                audioPlayer.play();
                audioPlayer.onended = () => {
                    isPlaying = false;
                    URL.revokeObjectURL(audioUrl);
                    playNextAudio();
                };
            }
        }

        recordButton.addEventListener('click', async () => {
            if (isRecording) {
                isRecording = false;
                recordButton.textContent = "Start Recording";
                recordButton.classList.remove('bg-red-600', 'hover:bg-red-700');
                recordButton.classList.add('bg-indigo-600', 'hover:bg-indigo-700');
                statusElement.textContent = "Stopping...";
                cancelButton.classList.add('hidden');
                if (socket) {
                    socket.emit('end_stream');
                    scriptProcessor.disconnect();
                    audioContext.close();
                }
            } else {
                try {
                    errorBox.classList.add('hidden');
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

                    audioContext = new AudioContext({ sampleRate: 16000 });
                    const source = audioContext.createMediaStreamSource(stream);
                    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

                    source.connect(scriptProcessor);
                    scriptProcessor.connect(audioContext.destination);

                    socket = io();
                    
                    socket.on('connect', () => {
                        console.log('Connected to server via WebSocket');
                        statusElement.textContent = "Recording...";
                        isRecording = true;
                        recordButton.textContent = "Stop Recording";
                        recordButton.classList.remove('bg-indigo-600', 'hover:bg-indigo-700');
                        recordButton.classList.add('bg-red-600', 'hover:bg-red-700');
                        cancelButton.classList.remove('hidden');
                        transcriptionOutput.innerHTML = '<p>Listening...</p>';
                        englishTranslationOutput.innerHTML = '<p></p>';
                        translationOutput.innerHTML = '<p></p>';

                        scriptProcessor.onaudioprocess = function(event) {
                            const inputBuffer = event.inputBuffer.getChannelData(0);
                            const float32Array = new Float32Array(inputBuffer);
                            socket.emit('audio_chunk', float32Array);
                        };
                    });

                    // Update this event handler to display all three text outputs
                    socket.on('transcription', (data) => {
                        console.log('Transcription received:', data);
                        transcriptionOutput.innerHTML = `<p>${data.original_text}</p>`;
                        englishTranslationOutput.innerHTML = `<p>${data.english_text}</p>`;
                        translationOutput.innerHTML = `<p>${data.translated_text}</p>`;
                        socket.emit('tts_request', { text: data.translated_text });
                    });
                    
                    socket.on('tts_response', (data) => {
                        const audioBlob = new Blob([new Uint8Array(data.audio_data)], { type: 'audio/mpeg' });
                        ttsAudioQueue.push(audioBlob);
                        playNextAudio();
                    });

                    socket.on('disconnect', () => {
                        console.log('Disconnected from server');
                        statusElement.textContent = "Disconnected";
                        scriptProcessor.disconnect();
                        audioContext.close();
                    });

                    socket.on('connect_error', (err) => {
                        console.error('Connection Error:', err);
                        displayError('Could not connect to the server. Please check the console for details.');
                    });
                    
                } catch (err) {
                    console.error('Microphone access denied:', err);
                    displayError('Microphone access was denied. Please allow microphone permissions in your browser settings.');
                }
            }
        });

        cancelButton.addEventListener('click', () => {
            if (isRecording) {
                isRecording = false;
                recordButton.textContent = "Start Recording";
                recordButton.classList.remove('bg-red-600', 'hover:bg-red-700');
                recordButton.classList.add('bg-indigo-600', 'hover:bg-indigo-700');
                statusElement.textContent = "Ready";
                cancelButton.classList.add('hidden');
                if (socket) {
                    socket.emit('cancel_stream');
                    scriptProcessor.disconnect();
                    audioContext.close();
                }
                resetOutputs();
            }
        });

        ttsButton.addEventListener('click', async () => {
            const text = ttsInput.value;
            if (text.trim() === '') {
                return;
            }

            ttsButton.disabled = true;
            ttsButton.textContent = "Synthesizing...";
            ttsStatusElement.classList.remove('hidden');

            try {
                const response = await fetch('/tts', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: text })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const audioBlob = await response.blob();
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);
                audio.play();
                audio.onended = () => {
                    ttsButton.disabled = false;
                    ttsButton.textContent = "Synthesize Speech";
                    ttsStatusElement.classList.add('hidden');
                };

            } catch (error) {
                console.error('TTS failed:', error);
            } finally {
                ttsButton.disabled = false;
                ttsButton.textContent = "Synthesize Speech";
                ttsStatusElement.classList.add('hidden');
            }
        });
    </script>
</body>
</html>
""")

@app.route('/tts', methods=['POST'])
def tts_synthesis():
    """Endpoint for Text-to-Speech synthesis."""
    if tts_tokenizer is None or tts_model is None:
        return {"error": "TTS model not loaded on server."}, 500
        
    data = request.json
    text = data.get('text', '')
    if not text:
        return {"error": "No text provided."}, 400

    try:
        inputs = tts_tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            speech_tensor = tts_model(**inputs).waveform
        
        speech_np = speech_tensor.cpu().numpy().squeeze()
        
        buffer = io.BytesIO()
        sf.write(buffer, speech_np, tts_model.config.sampling_rate, format='mp3')
        buffer.seek(0)
        
        return Response(buffer.read(), mimetype='audio/mpeg')

    except Exception as e:
        print(f"TTS error: {e}")
        return {"error": "An error occurred during TTS synthesis."}, 500

@socketio.on('tts_request')
def handle_tts_request(data):
    """Handles TTS requests from transcription, and sends audio back via WebSocket."""
    session_id = request.sid
    text = data.get('text', '')

    if not text:
        return
    
    if tts_tokenizer is None or tts_model is None:
        emit('tts_response', {'error': 'TTS model not loaded on server.'}, room=session_id)
        return

    try:
        inputs = tts_tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            speech_tensor = tts_model(**inputs).waveform
        
        speech_np = speech_tensor.cpu().numpy().squeeze()
        
        buffer = io.BytesIO()
        sf.write(buffer, speech_np, tts_model.config.sampling_rate, format='mp3')
        buffer.seek(0)
        
        emit('tts_response', {'audio_data': buffer.getvalue()}, room=session_id)

    except Exception as e:
        print(f"TTS error during real-time synthesis for session {session_id}: {e}")
        emit('tts_response', {'error': 'An error occurred during TTS synthesis.'}, room=session_id)


@socketio.on('connect')
def handle_connect():
    """Initializes the audio buffer for the new session."""
    session_id = request.sid
    session_data[session_id] = {
        'buffer': np.array([], dtype=np.float32)
    }
    print(f"Client connected with session ID: {session_id}")

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handles incoming raw audio chunks and appends them to the buffer."""
    session_id = request.sid
    
    if session_id not in session_data:
        return

    chunk = np.frombuffer(data, dtype=np.float32)
    session_data[session_id]['buffer'] = np.concatenate((session_data[session_id]['buffer'], chunk))

@socketio.on('end_stream')
def handle_end_stream():
    """Handles the end of the audio stream and processes the entire buffer."""
    session_id = request.sid
    
    if session_id not in session_data or len(session_data[session_id]['buffer']) == 0:
        emit('transcription', {
            'original_text': 'No audio recorded.',
            'english_text': '',
            'translated_text': ''
        }, room=session_id)
        if session_id in session_data:
            del session_data[session_id]
        print(f"Stream ended, no audio for session {session_id}")
        return
    
    if stt_processor is None or stt_model is None:
        emit('transcription', {'original_text': 'STT model not loaded on server.', 'english_text': '', 'translated_text': ''}, room=session_id)
        if session_id in session_data:
            del session_data[session_id]
        return

    try:
        audio_input = session_data[session_id]['buffer']
        
        # Step 1: STT (Gujarati speech to Gujarati text)
        input_values = stt_processor(
            audio_input, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        ).input_values
        
        logits = stt_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        gujarati_transcription = stt_processor.decode(predicted_ids[0], skip_special_tokens=True)
        print(f"STT Output (Devanagari): {gujarati_transcription}")
        
        # Transliterate from Devanagari to Gujarati script
        transliterated_gujarati_text = transliterate(
            gujarati_transcription,
            sanscript.DEVANAGARI,
            sanscript.GUJARATI
        )
        print(f"Transliterated Text (Gujarati): {transliterated_gujarati_text}")

        # Step 2: Gujarati Text to English Text
        if guj_en_translator:
            en_translation = guj_en_translator.translate(transliterated_gujarati_text)
        else:
            en_translation = "Translation model not loaded."
        
        print(f"English Translation: {en_translation}")
            
        # Step 2.5: Get response from AI assistant
        llm_response_en = get_llm_response(en_translation)
        print(f"AI Assistant Response (English): {llm_response_en}")

        # Step 3: English Text to Gujarati Text (round-trip)
        if en_guj_translator:
            gujarati_translation = en_guj_translator.translate(llm_response_en)
        else:
            gujarati_translation = "Translation model not loaded."

        print(f"Final Gujarati Translation: {gujarati_translation}")

        # Emit transcription results back to the client
        emit('transcription', {
            'original_text': transliterated_gujarati_text,
            'english_text': en_translation,
            'translated_text': gujarati_translation
        }, room=session_id)

    except Exception as e:
        print(f"Transcription/translation error for session {session_id}: {e}")
        emit('transcription', {
            'original_text': 'An error occurred during transcription.',
            'english_text': '',
            'translated_text': ''
        }, room=session_id)
    
    if session_id in session_data:
        del session_data[session_id]
    print(f"Stream ended, buffer for session {session_id} cleared.")

@socketio.on('cancel_stream')
def handle_cancel_stream():
    """Clears the buffer without processing."""
    session_id = request.sid
    if session_id in session_data:
        del session_data[session_id]
    print(f"Stream cancelled, buffer for session {session_id} cleared.")

@socketio.on('disconnect')
def handle_disconnect():
    """Cleans up the buffer when a client disconnects."""
    session_id = request.sid
    if session_id in session_data:
        del session_data[session_id]
    print(f"Client disconnected with session ID: {session_id}")

if __name__ == '__main__':
    print("Starting the real-time transcription server...")
    print("Please go to http://127.0.0.1:5001 in your web browser.")
    socketio.run(app, debug=True, port=5001)
    