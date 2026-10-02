"""
Simple Tamil + English Speech-to-Text app
-----------------------------------------
Uses faster-whisper for transcription and Gradio for the browser UI.
Everything runs locally on your machine.

Run with:  python app.py
Then open the link shown in the terminal (usually http://127.0.0.1:7860).
"""

import gradio as gr
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# 1. SETTINGS - change these if you like
# ---------------------------------------------------------------------------

# Model size: "tiny", "base", "small", "medium", "large-v3"
# Bigger = more accurate (especially for Tamil) but slower and larger download.
#   small    ~460 MB  -> good starting point, fast on most laptops
#   medium   ~1.5 GB  -> noticeably better Tamil accuracy
#   large-v3 ~3 GB    -> best accuracy, slow without a GPU
MODEL_SIZE = "small"

# "cpu" works everywhere. If you have an NVIDIA GPU, use "cuda" and
# change COMPUTE_TYPE to "float16" for a big speed-up.
DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # "int8" is fast and light on CPU

# Language choices shown in the dropdown -> Whisper language codes.
# None means "let Whisper detect the language automatically".
LANGUAGES = {
    "Auto-detect": None,
    "Tamil": "ta",
    "English": "en",
}

# ---------------------------------------------------------------------------
# 2. LOAD THE MODEL (once, when the app starts)
# ---------------------------------------------------------------------------
# The first run downloads the model, so it can take a few minutes.
print(f"Loading Whisper model '{MODEL_SIZE}' ... (first run downloads it)")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded.")


# ---------------------------------------------------------------------------
# 3. TRANSCRIPTION FUNCTION
# ---------------------------------------------------------------------------
def transcribe(audio_path, language_choice):
    """Takes an audio file path + chosen language, returns (text, info)."""

    # No audio uploaded or recorded yet
    if audio_path is None:
        return "Please upload an audio file or record from your microphone.", ""

    language_code = LANGUAGES[language_choice]  # e.g. "ta", "en", or None

    try:
        # transcribe() returns a generator of segments plus info about the audio
        segments, info = model.transcribe(
            audio_path,
            language=language_code,  # None = auto-detect
            beam_size=5,             # higher = slightly more accurate, slower
            vad_filter=True,         # skips silent parts, reduces hallucinations
        )

        # Join all the segments into one block of text
        text = " ".join(segment.text.strip() for segment in segments).strip()

        if not text:
            text = "(No speech detected in the audio.)"

        # Small status line showing which language was used/detected
        info_text = (
            f"Language: {info.language} "
            f"(confidence: {info.language_probability:.0%}) | "
            f"Duration: {info.duration:.1f}s"
        )
        return text, info_text

    except Exception as error:
        return f"Something went wrong: {error}", ""


# ---------------------------------------------------------------------------
# 4. BUILD THE GRADIO UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Tamil + English Speech-to-Text") as demo:
    gr.Markdown(
        "# 🎙️ Speech-to-Text (Tamil + English)\n"
        "Upload an audio file or record with your microphone, "
        "pick a language, and click **Transcribe**."
    )

    with gr.Row():
        with gr.Column():
            # Audio input: upload a file OR record from the mic
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",  # gives us a file path to pass to Whisper
                label="Audio (mp3, wav, etc.)",
            )
            language_input = gr.Dropdown(
                choices=list(LANGUAGES.keys()),
                value="Auto-detect",
                label="Language",
            )
            transcribe_button = gr.Button("Transcribe", variant="primary")

        with gr.Column():
            text_output = gr.Textbox(
                label="Transcription",
                lines=12,
                interactive=False,
            )
            info_output = gr.Textbox(label="Details", interactive=False)

    # Connect the button to our function
    transcribe_button.click(
        fn=transcribe,
        inputs=[audio_input, language_input],
        outputs=[text_output, info_output],
    )

# ---------------------------------------------------------------------------
# 5. START THE APP
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()  # opens on http://127.0.0.1:7860