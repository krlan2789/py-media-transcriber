import os
import time

os.environ["OMP_NUM_THREADS"] = "4"

import torch
import whisper
import faster_whisper

# device = 'cpu'
# if torch.cuda.is_available():
#     device = 'cuda'
# print("CUDA : " , torch.cuda.is_available())
# print("CUDA Current Device : " , torch.cuda.current_device())
# print("CUDA Is Initialized : " , torch.cuda.is_initialized())
# print("CUDA Memory Allocated : " , torch.cuda.memory_allocated())
# print("CUDA Memory Reserved : " , torch.cuda.memory_reserved())
# print("CUDA Memory Summary :\n" , torch.cuda.memory_summary())

def clear(text: str):
    return text.replace(',', r'').replace(' ', r'').replace('?', r'').replace('.', r'').replace('。', r'').replace('、', r'')

# Transcribe an audio file with openai whisper library
# model = "turbo" | "small" | "tiny"
def run_openai_whisper(audio, model: str, device = 'cpu'):
    if device.lower() == 'cuda' and torch.cuda.is_available() == False:
        device = 'cpu'

    model = whisper.load_model(model, device=device)
    result = model.transcribe(
        audio,
        beam_size=5,
        word_timestamps=True,
        language="zh",
        fp16=False,
        initial_prompt="CLC Mandarin",
        condition_on_previous_text=False,
    )
    return clear(result['text'])

# Transcribe an audio file with faster whisper library
# model = "large" | "small" | "large-v3" | "tiny"
def run_faster_whisper(audio, model: str, device = 'cpu'):
    if device.lower() == 'cuda' and torch.cuda.is_available() == False:
        device = 'cpu'

    model = faster_whisper.WhisperModel(model, device=device, compute_type="int8")
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        word_timestamps=True,
        language="zh",
        initial_prompt="CLC Mandarin",
    )
    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    text = ""
    sentence = ""
    for segment in segments:
        text += segment.text + ' '
        sentence += ' '
        for word in segment.words:
            sentence += word.word
    return clear(text)

# Transcribe multiple audio files with faster whisper library
def transcribeFasterWhisper(files: [], model: str, device = 'cpu'):
    results = ""
    totalTime = 0
    for path in files:
        start = time.time()
        result = run_faster_whisper(path, model, device)
        results += result
        end = time.time()
        timeElapsed = end - start
        totalTime += timeElapsed
        print("\nfaster-whisper[%s]: %f seconds\n%s" % (model, timeElapsed, result))
    return results, totalTime

# Transcribe multiple audio files with openai whisper library
def transcribeOpenaiWhisper(files: [], model: str, device = 'cpu'):
    results = ""
    totalTime = 0
    for path in files:
        start = time.time()
        result = run_openai_whisper(path, model, device)
        results += result
        end = time.time()
        timeElapsed = end - start
        totalTime += timeElapsed
        print("\nopenai-whisper[%s]: %f seconds\n%s" % (model, timeElapsed, result))
    return results, totalTime

# ------------------------------------------------------------------------------------ #

# File to transcribe
audioPath = "audio/W07D4.m4a"
path = './audio/082RDD1/'
audiosPath = os.listdir(path)
for idx, f in enumerate(audiosPath):
    audiosPath[idx] = path + audiosPath[idx]

# File to compare
textPath = "./audio/W07D4.txt"
textContent = ""

if (os.path.isfile(textPath)):
    textFile = open(textPath, "r", encoding="utf8")
    textContent = textFile.read()
    textFile.close()
    # print("->\n" + textContent)

def transcribe(files: list):
    model = "large-v3"
    device = 'cuda'
    result, totalTime = transcribeFasterWhisper(files, model, device)
    print(
        "\n\n------------------------------------\nWith Faster-Whisper ->\nDevice     : %s\nModel Size : %s\nTotal Time : %f seconds\nTranscribed: %s\nTextContent: %s\n------------------------------------\n\n"
        % (device, model, totalTime, result, textContent)
    )
    model = "large-v3"
    device = 'cuda'
    result, totalTime = transcribeFasterWhisper(files, model, device)
    print(
        "\n\n------------------------------------\nWith Faster-Whisper ->\nDevice     : %s\nModel Size : %s\nTotal Time : %f seconds\nTranscribed: %s\nTextContent: %s\n------------------------------------\n\n"
        % (device, model, totalTime, result, textContent)
    )

    # model = "base"
    # result, totalTime = transcribeOpenaiWhisper(files, model)
    # print("\n\n------------------------------------\nWith OpenAI-Whisper ->\nModel Size : %s\nTotal Time : %f seconds\nTranscribed: %s\nTextContent: %s\n------------------------------------\n\n" % (model, totalTime, result, textContent))

transcribe(audiosPath)