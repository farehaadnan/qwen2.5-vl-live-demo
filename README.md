# Qwen 2.5 VL(3B)

A set of Colab notebook for experimenting with **Qwen2.5-VL-3B-Instruct**, an open-source vision-language model, across three use cases: single-image Q&A, full video understanding, and a live webcam assistant with voice input and spoken answers.

## Features

1. **Image Q&A** - Ask a question about a single image or webcam snapshot (e.g. "What am I holding in my hand?") and get a text answer.
2. **Video description** - Upload a video file and get a description of what happens in it, using Qwen2.5-VL's native video/temporal understanding.
3. **Live webcam assistant** - Keeps your webcam running, listens for a spoken question, transcribes it locally with Whisper, sends the current frame + question to the model, and reads the answer back out loud. Runs in a loop until stopped.

## How it works

All three features use the same underlying model and a shared `ask_about_photo` / `ask_about_video` style function that builds a chat message with an image or video plus a text prompt, then runs it through Qwen2.5-VL.

- **Image Q&A** sends a single frame and a question to the model.
- **Video description** samples frames from an uploaded video at a set FPS and passes them to the model together, so it can reason about sequence and timing.
- **Live assistant** combines webcam frame capture, audio recording, Whisper transcription, and browser text-to-speech into one continuous loop.

This is snapshot- and clip-based reasoning, not continuous real-time video understanding. See Limitations below.

## Setup

Open the notebook in Google Colab. A GPU runtime is required: Runtime > Change runtime type > T4 GPU (or better).

### Install dependencies

```bash
pip install -q transformers accelerate qwen-vl-utils[decord] av openai-whisper
```

### Load the model

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)
```

## Usage

### 1. Image Q&A

Run the image cell, pass in a file path or a webcam snapshot, and ask a question:

```python
answer = ask_about_photo("your_image.jpg", "What am I holding in my hand?")
print(answer)
```

### 2. Video description

Upload a video file to the Colab file browser (or use `google.colab.files.upload()`), then run:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "your_video.mp4", "fps": 1.0},
            {"type": "text", "text": "Describe what happens in this video."},
        ],
    }
]
```

Lower the `fps` value for longer videos to avoid running out of memory.

### 3. Live webcam assistant

Run the full assistant cell. It will:

- Start the webcam and keep it running.
- Record a few seconds of audio when prompted.
- Transcribe the audio locally with Whisper.
- Send the current frame and transcribed question to the model.
- Print the answer and read it out loud.

On first run, your browser will ask for camera and microphone permissions - accept both. Click the stop button on the cell at any time to end the loop and release the camera.

## Configuration

| Setting | Where | Notes |
|---|---|---|
| Model size | `model_id` | Use `Qwen/Qwen2.5-VL-3B-Instruct` for T4 GPUs; `Qwen/Qwen2.5-VL-7B-Instruct` if you have an A100/L4 with more VRAM |
| Video sampling rate | `fps` in the video message | Lower for longer videos to save memory |
| Whisper model | `whisper.load_model("base")` | Use `"tiny"` for faster, less accurate transcription |
| Recording length | `record_audio(seconds=4)` | Increase if questions get cut off |
| Answer length | `max_new_tokens` | Lower for faster responses, higher for more detail |

## Limitations

- Not continuous real-time video - the live assistant reasons over single snapshots taken a few seconds apart, not a continuous feed. Fast changes between captures can be missed.
- Latency - each cycle in the live assistant (record, transcribe, infer, speak) takes roughly 5 to 8 seconds on a free T4 GPU.
- Video length and memory - longer videos need a lower `fps` or reduced resolution to avoid out-of-memory errors, especially on a T4.
- Browser speech-to-text (`SpeechRecognition`) is unreliable inside Colab's sandboxed output frame, which is why this project uses raw audio capture plus Whisper instead.
- Session-based - camera and microphone state resets if the Colab runtime restarts or the browser tab is refreshed.

## Acknowledgments

- Qwen2.5-VL by Alibaba
- Whisper by OpenAI
