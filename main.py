import os
import sounddevice as sd
from scipy.io.wavfile import write

import whisper
import torch
import numpy as np

import openai
from dotenv import load_dotenv
load_dotenv()

import pyperclip

# transcribe audio file
def transcribe():
    torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = whisper.load_model("tiny", device=DEVICE)
    print(
        f"Model is {'multilingual' if model.multilingual else 'English-only'}"
        f"and has {sum(np.prod(p.shape) for p in model.parameters())} parameters."
    )
    audio = whisper.load_audio("output.mp3")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print("Detected languages: {max(probs, key=probs.get)}")

    options = whisper.DecoderOptions(language="en",without_timestamps=True,fp16 = True)
    result = whisper.decode(model, mel, options)
    print(result["text"])
    return result["text"]
    
# record audio
def record(duration):
    fs=44100 # sample rate
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    print("Recording Audio - Speak now")
    sd.wait() # wait until recording is finished
    print("Audio recording complete")
    write("output.mp3", fs, myrecording) # save as MP3 file

# generate email with gpt3
def generate_email(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Create a kind and formal email for this reason: {text}",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response["choices"][0]["text"]

def main():
    record(8)
    text = transcribe()
    email = generate_email(text)
    print(email)
    pyperclip.copy(email)

if __name__ == "__main__":
    main()

