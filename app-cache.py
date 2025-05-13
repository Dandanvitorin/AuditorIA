
import streamlit as st
from main import analisar_audio
import re

st.set_page_config(page_title="Análise de Áudio", layout="wide")
st.title("🧠 Plataforma de Análise de Áudio com IA")

aba = st.sidebar.radio("Funcionalidade", ["Transcrição", "Classificação de Emoções", "Palavras Frequentes", "Frases Proibidas"])

audio_file = st.file_uploader("📂 Envie o arquivo de áudio (formato .wav ou .mp3)", type=["wav", "mp3"])

@st.cache_data(show_spinner="🔍 Processando o áudio...")
def processar_arquivo(audio_file):
    return analisar_audio(audio_file)

def exibir_transcricao_formatada(transcricoes):
    for frase in transcricoes:
        match = re.match(r"\[(SPEAKER_\d+)\]\s*(.*)", frase)
        if match:
            speaker, texto = match.groups()
            st.markdown(f"**{speaker}:** {texto}")
        else:
            st.markdown(frase)

if audio_file:
    resultado = processar_arquivo(audio_file)

    if aba == "Transcrição":
        st.subheader("📃 Transcrição do Áudio")
        exibir_transcricao_formatada(resultado["transcricao"])

    elif aba == "Classificação de Emoções":
        st.subheader("😊 Emoções no Áudio")
        st.dataframe(resultado["emocao"])

    elif aba == "Palavras Frequentes":
        st.subheader("🔤 Análise de Palavras Frequentes")
        st.pyplot(resultado["wordcloud_fig"])
        st.dataframe(resultado["tabela_palavras"])

    elif aba == "Frases Proibidas":
        st.subheader("🚫 Frases Proibidas Detectadas")
        proibidas = resultado["proibidas"]
        if proibidas:
            st.error("⚠️ Frases Proibidas Encontradas:")
            for frase in proibidas:
                st.markdown(f"- {frase}")
        else:
            st.success("✅ Nenhuma frase proibida encontrada.")
else:
    st.info("Por favor, envie um arquivo de áudio para começar.")