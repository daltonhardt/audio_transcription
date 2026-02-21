import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os
import time
from pydub import AudioSegment

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_cached_model(model_size):
    model = WhisperModel(
        model_size_or_path=model_size,  # tiny / base / small / medium / large-v3 / turbo
        device="cpu",                   # use "cuda" se tiver GPU
        compute_type="int8",            # economiza RAM
        cpu_threads=os.cpu_count(),     #
        num_workers=4                   # paralelismo interno
    )
    return model


# ---------- SPLIT AUDIO ----------
def split_audio(audio_path, chunk_ms=5000):
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_ms):
        chunk = audio[i:i+chunk_ms]
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp.name, format="wav")
        chunks.append(temp.name)

    return chunks


# ---------- ULTRA FAST TRANSCRIPTION ----------
def transcribe_stream(model, audio_file, output_box):
    tic = time.perf_counter()
    # salva áudio original
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.getbuffer())
        temp_path = tmp.name

    # divide em pedaços
    chunks = split_audio(temp_path, chunk_ms=6000)

    final_text = ""
    for chunk in chunks:
        segments, info = model.transcribe(
            chunk,
            beam_size=1,
            language="pt",
            vad_filter=True,
            condition_on_previous_text=False
        )
        partial = "".join([seg.text for seg in segments])
        final_text += partial + " "
        # atualização em tempo real
        output_box.markdown(f":green-background[{final_text}]")

    toc = time.perf_counter()
    st.write(f'Tempo: {round(toc - tic, 1)} s')
    return final_text


# ---------- UI ----------
st.subheader("Transcrição de Áudio - Faster Whisper")
lista_modelos = ['tiny', 'base', 'small', 'medium', 'large']
modelo = st.selectbox("Escolha o modelo: ", lista_modelos, index=1)
model = load_cached_model(modelo)

# para gravar um áudio com a mensagem
audio_value = st.audio_input("Grave uma message")
if audio_value:
    # if st.button("Transcrição ULTRA FAST"):
    # st.audio(audio_value)
    output_box = st.empty()
    with st.spinner("Transcrevendo...", show_time=True):
        texto = transcribe_stream(model, audio_value, output_box)

# para carregar um arquivo com o áudio da mensagem
audio_file = st.file_uploader("Selecione um arquivo", type=["mp3", "wav", "m4a"])
if audio_file:
    output_box = st.empty()
    with st.spinner("Transcrevendo...", show_time=True):
        texto = transcribe_stream(model, audio_file, output_box)

