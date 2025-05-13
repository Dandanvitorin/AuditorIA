import whisper
from typing import Dict, List
from pydub import AudioSegment
from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv
import torch
import gc

load_dotenv()

# Carrega o modelo Whisper uma vez
_modelo_whisper = whisper.load_model("medium")

# Define o pipeline da diarização
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=os.getenv("HUGGINGFACE_API_KEY")
)


def diarizar_audio(caminho_audio: str):
    return pipeline(caminho_audio)


def cortar_segmentos(caminho_audio: str, diarization) -> List[Dict]:
    audio = AudioSegment.from_file(caminho_audio)
    segmentos = []

    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        inicio = int(turn.start * 1000)
        fim = int(turn.end * 1000)
        trecho = audio[inicio:fim]

        nome_arquivo = f"segmento_{i:02d}_{speaker}.wav"
        trecho.export(nome_arquivo, format="wav")

        segmentos.append({
            "arquivo": nome_arquivo,
            "speaker": speaker,
            "inicio": turn.start,
            "fim": turn.end
        })

    return segmentos


def transcrever_segmentos(segmentos: List[Dict]) -> List[str]:
    transcricoes = []
    for seg in segmentos:
        result = _modelo_whisper.transcribe(seg["arquivo"], language="pt")
        texto = result["text"].strip()
        transcricoes.append(f"[{seg['speaker']}] {texto}")
        torch.cuda.empty_cache()
        gc.collect()
    return transcricoes


def transcrever_com_diarizacao(caminho_audio: str) -> None:
    print("⏳ Diarizando falantes...")
    diarizacao = diarizar_audio(caminho_audio)

    print("✂️ Cortando segmentos...")
    segmentos = cortar_segmentos(caminho_audio, diarizacao)

    print("📝 Transcrevendo...")
    transcricoes = transcrever_segmentos(segmentos)
    return transcricoes

    print("\n🎤 Transcrição final:\n")
    for t in transcricoes:
        print(t)




# Teste local
if __name__ == "__main__":
    caminho = '/home/daniel/Downloads/WhatsApp-Ptt-2025-04-29-at-09.55.04.mp3'
    transcrever_com_diarizacao(caminho)
