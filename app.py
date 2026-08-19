import streamlit as st
from faster_whisper import WhisperModel


# --------------------------------------------------
# Configuración
# --------------------------------------------------

st.set_page_config(
    page_title="Audio → Texto",
    page_icon="🎙️",
    layout="wide"
)


# --------------------------------------------------
# Cargar modelo
# --------------------------------------------------

@st.cache_resource
def cargar_modelo():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )


model = cargar_modelo()


# --------------------------------------------------
# Función de transcripción
# --------------------------------------------------

def transcribir(audio):

    if audio is None:
        return ""

    segments, info = model.transcribe(
        audio,
        language="es",
        vad_filter=True
    )

    texto = " ".join(
        segment.text.strip()
        for segment in segments
    )

    return texto


# --------------------------------------------------
# Interfaz
# --------------------------------------------------

st.title("🎙️ Audio → Texto")

st.markdown(
    """
    Convierte archivos de audio a texto automáticamente.

    Demo proporcionada por el equipo de Michigeeks.
    """
)

st.divider()


col1, col2 = st.columns(2)


# --------------------------------------------------
# Audio
# --------------------------------------------------

with col1:

    st.subheader("🎧 Audio")

    audio = st.file_uploader(
        "Sube un archivo de audio",
        type=[
            "mp3",
            "wav",
            "m4a",
            "ogg",
            "flac",
            "aac"
        ]
    )

    if audio is not None:

        st.audio(
            audio,
            format=audio.type
        )

        transcribir_btn = st.button(
            "🚀 Transcribir",
            type="primary",
            use_container_width=True
        )

    else:

        transcribir_btn = False


# --------------------------------------------------
# Resultado
# --------------------------------------------------

with col2:

    st.subheader("📝 Transcripción")

    if transcribir_btn:

        with st.spinner("Transcribiendo audio..."):

            # Guardar temporalmente el archivo
            import tempfile

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".audio"
            ) as temp:

                temp.write(audio.getbuffer())
                temp_path = temp.name

            texto = transcribir(temp_path)

        if texto:

            st.text_area(
                "Resultado",
                value=texto,
                height=350
            )

        else:

            st.warning(
                "No se pudo obtener texto del audio."
            )

    else:

        st.info(
            "Sube un audio y pulsa «Transcribir»."
        )