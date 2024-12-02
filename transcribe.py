import os
import time

os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import json
import torch

# import whisper
import faster_whisper


def clear(text: str):
    return (
        text.replace(" ", r"")
        .replace(",", r"")
        .replace("?", r"")
        .replace(".", r"")
        .replace("。", r"")
        .replace("、", r"")
    )


# # Transcribe an audio file with openai whisper library
# # model = "turbo" | "small" | "tiny"
# def run_openai_whisper(audio, model: str, device="cpu"):
#     if device.lower() == "cuda" and torch.cuda.is_available() == False:
#         device = "cpu"

#     model = whisper.load_model(model, device=device)
#     result = model.transcribe(
#         audio,
#         beam_size=5,
#         word_timestamps=True,
#         language="zh",
#         fp16=False,
#         initial_prompt="CLC Mandarin",
#         condition_on_previous_text=False,
#     )
#     return clear(result["text"])


# # Transcribe multiple audio files with openai whisper library
# def transcribeOpenaiWhisper(files: [], model: str, device="cpu"):
#     results = ""
#     totalTime = 0
#     for path in files:
#         start = time.time()
#         result = run_openai_whisper(path, model, device)
#         print("transcribeOpenaiWhisper.result: " + result)
#         results += result
#         end = time.time()
#         timeElapsed = end - start
#         totalTime += timeElapsed
#         # print("\nopenai-whisper[%s]: %f seconds\n%s" % (model, timeElapsed, result))
#     return results, totalTime


# Transcribe an audio file with faster whisper library
# model = "large" | "small" | "large-v3" | "tiny"
def run_faster_whisper(audio, model: str, device="cpu"):
    if device.lower() == "cuda" and torch.cuda.is_available() == False:
        device = "cpu"

    model = faster_whisper.WhisperModel(model, device=device, compute_type="int8")
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        word_timestamps=True,
        language="zh",
        initial_prompt="CLC Mandarin",
    )
    # print(
    #     "Detected language '%s' with probability %f%%"
    #     % (info.language, (info.language_probability * 100))
    # )
    text = ""
    sentence = ""
    for segment in segments:
        text += segment.text + " "
        sentence += " "
        for word in segment.words:
            sentence += word.word

    return text.strip()  # clear(text)


# Transcribe multiple audio files with faster whisper library
def transcribeFasterWhisper(files: [], model: str, device="cpu"):
    results = []
    totalTime = 0
    for path in files:
        start = time.time()
        # print("run_faster_whisper(executed): " + path)
        result = run_faster_whisper(path, model, device)
        # print("transcribeFasterWhisper.result: " + result)
        results.append(result)
        end = time.time()
        timeElapsed = end - start
        totalTime += timeElapsed
        # print("\nfaster-whisper[%s]: %f seconds\n%s" % (model, timeElapsed, result))

    # print("\nfaster-whisper[%s]: %s" % (model, results))
    return results, totalTime


# ------------------------------------------------------------------------------------ #


def transcribe(files: list, model="base", device="cpu", textContent=""):
    results, totalTime = transcribeFasterWhisper(files, model, device)
    # print(
    #     "\n\n------------------------------------\nWith Faster-Whisper ->\nDevice     : %s\nModel Size : %s\nTotal Time : %f seconds\nTranscribed: %s\nTextContent: %s\n------------------------------------\n\n"
    #     % (device, model, totalTime, results, textContent)
    # )
    return {
        "status": True,
        "device": device,
        "model_size": model,
        "time_elaped": totalTime,
        "transcription": results,
    }


# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument("--audiodir", help="Audio Directory", default="")
parser.add_argument("-a", nargs="+", help="Audio files path", default=[])
parser.add_argument("-m", help="Model size: tiny|small|large|large-v3", default="small")
parser.add_argument("-d", help="Device: cpu|cuda", default="cpu")
args = parser.parse_args()

audioFileDir = args.audiodir
audioFilesPath = args.a
modelSize = args.m
device = args.d

results = {
    "status": False,
}

if __name__ == "__main__":
    if audioFileDir or audioFilesPath:
        audiosPath = []

        # Getting audio files to transcribe
        if audioFileDir:
            audiosPath = os.listdir(audioFileDir)
            for idx, f in enumerate(audiosPath):
                audiosPath[idx] = audioFileDir + audiosPath[idx]

        if audioFilesPath:
            for idx, f in enumerate(audioFilesPath):
                audiosPath.append(f)
        # print(f"Audio files Path  : {audiosPath}")

        # print("\nTranscription audio files..")
        results = transcribe(audiosPath, modelSize, device)
        # print("Audio files transcribed!")

print(json.dumps(results, ensure_ascii=False))
