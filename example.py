"""Script for diarization and transcription
"""
from pathlib import Path
from typing import Union
from pyannote.audio import Pipeline
import wave
import csv
import numpy as np
import tempfile
import whisper


def divide_wav_file(file_path: Union[str, Path],
                    start_time: float, end_time: float):
    """Takes slice of the wav file given start and end time

    Args:
        file_path (Union[str, Path]): path to wav file
        start_time (float): start time
        end_time (float): end time

    """
    with wave.open(file_path, 'rb') as wav_file:
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()

        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)

        # Set the file pointer to the starting frame
        wav_file.setpos(start_frame)

        # Read the frames for the specified portion
        frames = wav_file.readframes(end_frame - start_frame)

    # Convert the byte data to numpy array
    audio_array = np.frombuffer(frames, dtype=np.int16)

    # Save audio data to temporary .wav file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        with wave.open(temp_wav.name, 'wb') as temp_file:
            temp_file.setnchannels(1)  # Mono audio
            temp_file.setsampwidth(sample_width)
            temp_file.setframerate(frame_rate)
            temp_file.writeframes(frames)

        temp_wav_path = temp_wav.name

    return audio_array, frame_rate, temp_wav_path


model = whisper.load_model("large")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token="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")

input_file = "Long.wav"
diarization = pipeline(input_file)


transcribed = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    _, _, temp_wav_path = divide_wav_file(input_file, turn.start, turn.end)
    text = model.transcribe(temp_wav_path)['text']
    transcribed.append({
        'speaker': speaker,
        'text': text
    })

with open('transcribed.csv', 'w', encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['Speaker', 'Text'])
    for line in transcribed:
        writer.writerow([line['speaker'], line['text']])
