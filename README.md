# Automatic-speech-recognition
# 🎙️ Tamil + English Speech-to-Text

<img width="1896" height="805" alt="1" src="https://github.com/user-attachments/assets/7f8560e8-85d8-4462-bbc8-b8026d45f0e7" />
<img width="1897" height="817" alt="2" src="https://github.com/user-attachments/assets/57dc85ce-440a-43c9-838e-7616d167d573" />




## Features

- Upload an audio file (mp3, wav, etc.) or record from your microphone
- Choose **Tamil**, **English**, or **Auto-detect**
- Shows the transcription plus detected language and audio duration
- Single file (`app.py`), easy to read and modify

## Requirements

- Python 3.9 or newer
- (Recommended) [FFmpeg](https://ffmpeg.org/) for broader audio format support

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create a virtual environment (recommended)**

   Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   Mac / Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg (optional but recommended)**
   - Windows: `winget install ffmpeg` (then reopen the terminal)
   - Mac: `brew install ffmpeg`

## Run

```bash
python app.py
```

Open the link shown in the terminal (usually http://127.0.0.1:7860).

> The first run downloads the Whisper model (~460 MB for `small`). It is cached afterwards.

## Configuration

Edit the settings at the top of `app.py`:

| Setting | Default | Notes |
|---|---|---|
| `MODEL_SIZE` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`. Larger = better Tamil accuracy but slower. |
| `DEVICE` | `cpu` | Use `cuda` if you have an NVIDIA GPU. |
| `COMPUTE_TYPE` | `int8` | Use `float16` with `cuda`. |

## Tips

- Tamil accuracy improves noticeably with `medium` or `large-v3`.
- Auto-detect picks one language for the whole clip. For mixed-language audio, select the main language manually.
- Your browser will ask for microphone permission the first time you record.

## Troubleshooting

**`TypeError: argument of type 'bool' is not iterable` / "When localhost is not accessible..."**
A Gradio/pydantic version mismatch. Fix with:
```bash
pip install --upgrade gradio gradio_client
```
If it persists:
```bash
pip install pydantic==2.10.6
```

**Symlink warning on Windows** — harmless; the model still works.

## Built with

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Gradio](https://www.gradio.app/)
- OpenAI Whisper models
