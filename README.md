# Python Media Transcriber

Media Transcriber using Faster Whisper Model

## Requirements

- Python v3.10 (3.12 recommended)
- faster-whisper v1.0.3
- torch v2.4.1
- torchaudio v2.4.1

## Run Basic Command

``` shell
python transcribe.py -a 'D:/Files/Documents/Python/Projects/py-media-transcriber/audio/1.m4a' 'D:/Files/Documents/Python/Projects/py-media-transcriber/audio/2.m4a'
```

## Help

```shell
--audiodir, Audio directory,                    to transcribe all audio files inside the directory
-a,         Audio files path,                   audio file path to transcribe (passible pass multiple file path)
-m,         LLM Model size,                     Available: tiny|small|large|large-v3 (default=small)
-d,         Device to perform transcription,    Available: cpu|cuda (default=cpu)
```