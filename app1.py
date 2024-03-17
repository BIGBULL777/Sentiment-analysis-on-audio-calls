import pyaudio
import gradio as gr
import whisper
from transformers import pipeline

model = whisper.load_model("base")
sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")

def analyze_sentiment(text):
    results = sentiment_analysis(text)
    sentiment_results = {result['label']: result['score'] for result in results}
    return sentiment_results

def inference(audio,sentiment_option):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5

    audio_data = []
    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        audio_data.append(in_data)
        return (in_data, pyaudio.paContinue)

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback)

    stream.start_stream()

    try:
        while True:
            pass  # Stream audio continuously, perform sentiment analysis as needed
    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    p.terminate()

title = """<h1 align="center">🎤 Multilingual ASR 💬</h1>"""
# Define other variables and UI components as needed...
custom_css = """
#banner-image {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
#chat-message {
    font-size: 14px;
    min-height: 300px;
}
"""
description = """ 
"""
block = gr.Blocks(css=custom_css)

with block:
    gr.HTML(title)

    with gr.Row():
        with gr.Column():
            gr.Image('thmbnail.jpg', elem_id="banner-image", show_label=False)
        with gr.Column():
            gr.HTML(description)

    with gr.Group():
        with gr.Group():
            audio = gr.Audio(
                label="Input Audio",
                show_label=False,
                sources="microphone",
                type="filepath"
            )

            sentiment_option = gr.Radio(
                choices=["Sentiment Only", "Sentiment + Score"],
                label="Select an option",
                # default="Sentiment Only"
            )

            btn = gr.Button("Transcribe")

        lang_str = gr.Textbox(label="Language")

        text = gr.Textbox(label="Transcription")

        sentiment_output = gr.Textbox(label="Sentiment Analysis Results",
                                    #    output=True
                                       )

        btn.click(inference, inputs=[audio, sentiment_option], outputs=[lang_str, text, sentiment_output])

       

block.launch()

