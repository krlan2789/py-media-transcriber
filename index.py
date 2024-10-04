import whisper

model = whisper.load_model("base")
result = model.transcribe("path/to/your/audio/file.wav")
print(result["text"])
