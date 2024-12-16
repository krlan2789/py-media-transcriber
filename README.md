# Python Media Transcriber

Media Transcriber using Faster Whisper Model

## Requirements

- Python v3.10 (3.12 recommended)
- faster-whisper v1.0.3

## Run Basic Command

``` shell
python transcribe.py -a 'D:/Files/Documents/Python/Projects/py-media-transcriber/audio/1.m4a' 'D:/Files/Documents/Python/Projects/py-media-transcriber/audio/2.m4a'
```

Example results

```json
{
    "status": true,
    "device": "cpu",
    "model_size": "large",
    "time_elaped": 18.026545763015747,
    "transcription": [
        "7-2-1",
        "2"
    ]
}
```

## Help

```shell
usage: transcribe.py [-h] [--audiodir AUDIODIR] [-a A [A ...]] [-m M] [-d D]

options:
  -h, --help           show this help message and exit
  --audiodir AUDIODIR  Audio Directory
  -a A [A ...]         Audio files path
  -m M                 Model size: tiny|small|large|large-v3
  -d D                 Device: cpu|cuda
```


# Python Text Similarity

## Run Basic Command

```shell
python similarity.py -t "A journey of a thousand miles begins with a single step" -s "A journey of a thousand miles begins with a single leap"
```

Example results

```json
{
    "status": true,
    "similarity": "94.5%",
    "transcription": "A journey of a thousand miles begins with a single step",
    "original": "A journey of a thousand miles begins with a single leap"
}
```

## Help

```shell
usage: similarity.py [-h] [-t T] [-s S]

options:
  -h, --help  show this help message and exit
  -t T        Transcription results
  -s S        Original sample text
```