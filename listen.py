from collections import deque
import re
from time import time, sleep

import faster_whisper
import numpy as np
import pyaudio
import torch
from gpiozero import AngularServo

device = "cpu"
torch.device(device)

ALPHA_PATTERN = re.compile('[\W_]+')
VAD_WINDOW_SIZE = 4096  # Window size for voice detection model
CHUNK_BUFFER = deque()  # Records chunks of audio each of length VAD_WINDOW_SIZE
CHUNK_SIZE_SECONDS = 1.5  # Maximum command duration (Determines BUFFER_SIZE)
POST_PAUSE_THRESH = 0.1  # Required time of no speech to be considered the end of a statement
RATE = 16000  # This must be 16kHz as this is what the models were trained on
REFRACTORY = 2  # Minimum time between commands
CHUNK_SIZE = int(RATE * CHUNK_SIZE_SECONDS)
BUFFER_SIZE=int(CHUNK_SIZE / VAD_WINDOW_SIZE)
CPU_THREADS=4
VAD_MODEL = 'snakers4/silero-vad'
COMMANDS = {
    "be gone shit": "flush",
    "be gone": "flush",
    "flush": "flush",
}


def clean(string):
    return ALPHA_PATTERN.sub(" ", string.lower()).strip()

def chunk_2_input(chunk):
    return (np.frombuffer(chunk, np.int16) / 32768).astype(np.float32)

def chunk_2_torch(chunk):
    return torch.from_numpy(chunk_2_input(chunk))

def buffer_2_input():
    return chunk_2_input(b''.join(CHUNK_BUFFER))

# vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',model='silero_vad',force_reload=True,onnx=False)
vad_model = torch.jit.load('silero_vad.pth').eval().to(device)
transcription_model = faster_whisper.WhisperModel("tiny.en", device=device, compute_type="float32", cpu_threads=CPU_THREADS)
streamIn = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, output=True, frames_per_buffer=VAD_WINDOW_SIZE)
servo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)

t0 = t = time()
i=0
last_activation = t0
print("Looping")
speech_start = 0
speech_end = 0
listening=False
while True:
    t = time()
    chunk = streamIn.read(VAD_WINDOW_SIZE)
    CHUNK_BUFFER.append(chunk)
    if len(CHUNK_BUFFER) > BUFFER_SIZE:
        CHUNK_BUFFER.popleft()
    
    if ((t-last_activation) < REFRACTORY):
        continue

    voice = vad_model(chunk_2_torch(chunk), RATE).item() > 0.5
    if voice:
        speech_end = t
        if not listening:
            listening = True
            speech_start = t
    else:
        if listening and (t-speech_end) > POST_PAUSE_THRESH:
            listening = False
            if (speech_end - speech_start) < CHUNK_SIZE_SECONDS:
                segments, info = transcription_model.transcribe(
                    buffer_2_input(), 
                    # beam_size=5,
                    language="en", 
                    condition_on_previous_text=False,
                    word_timestamps=False,
                )

                for segment in segments:
                    command = clean(segment.text)
                    if command in COMMANDS:
                        print(f"FLUSH '{command}'")
                        servo.angle = 90
                        sleep(1)
                        servo.angle = 0
                        last_activation = t
                    else:
                        print(f"UNKNOWN '{command}'")
