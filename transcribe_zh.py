from transformers import pipeline

transcriber = pipeline(
    "automatic-speech-recognition",
    model="jonatasgrosman/whisper-large-zh-cv11"
)

transcriber.model.config.forced_decoder_ids = (
    transcriber.tokenizer.get_decoder_prompt_ids(
        language="zh",
        task="transcribe"
    )
)

audioPath = "audio/W07D4.m4a"
transcription = transcriber(audioPath)
