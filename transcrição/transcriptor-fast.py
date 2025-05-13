import whisper
import os
import subprocess
from dotenv import load_dotenv
from typing import Dict, List
from pyannote.audio import Pipeline
from concurrent.futures import ThreadPoolExecutor
import librosa

# Carrega variáveis de ambiente
load_dotenv()

# Define o pipeline da diarização (usa token do .env)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.getenv("HUGGINGFACE_API_KEY")
)


def diarizar_audio(caminho_audio: str):
    return pipeline(caminho_audio)


def cortar_segmentos_ffmpeg(caminho_audio: str, diarization) -> List[Dict]:
    segmentos = []

    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        inicio = turn.start
        duracao = turn.end - turn.start

        if duracao < 0.5:
            continue  # ignora segmentos muito curtos

        nome_arquivo = f"segmento_{i:02d}_{speaker}.wav"

        cmd = [
            "ffmpeg", "-y",
            "-i", caminho_audio,
            "-ss", str(inicio),
            "-t", str(duracao),
            "-acodec", "copy",
            nome_arquivo
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        segmentos.append({
            "arquivo": nome_arquivo,
            "speaker": speaker,
            "inicio": inicio,
            "fim": turn.end
        })

    return segmentos


def transcrever_segmentos(segmentos: List[Dict]) -> List[str]:
    def transcreve(seg):
        try:
            y, sr = librosa.load(seg["arquivo"], sr=None)
            duracao = librosa.get_duration(y=y, sr=sr)
            if duracao < 0.5:
                return f"[{seg['speaker']}] (segmento muito curto ou silencioso)"
        except:
            return f"[{seg['speaker']}] (erro ao carregar o áudio)"

        # ✅ Carrega modelo isolado por thread para evitar erro de cache
        modelo = whisper.load_model("medium")
        result = modelo.transcribe(seg["arquivo"], language="pt")
        texto = result["text"].strip()
        return f"[{seg['speaker']}] {texto}"

    with ThreadPoolExecutor(max_workers=2) as executor:
        transcricoes = list(executor.map(transcreve, segmentos))

    return transcricoes


def transcrever_com_diarizacao(caminho_audio: str) -> None:
    print("⏳ Diarizando falantes...")
    diarizacao = diarizar_audio(caminho_audio)

    print("✂️ Cortando segmentos com ffmpeg...")
    segmentos = cortar_segmentos_ffmpeg(caminho_audio, diarizacao)

    print("📝 Transcrevendo com Whisper (paralelo)...")
    transcricoes = transcrever_segmentos(segmentos)

    print("\n🎤 Transcrição final:\n")
    for t in transcricoes:
        print(t)

    print("\n🧹 Limpando arquivos temporários...")
    for seg in segmentos:
        os.remove(seg["arquivo"])


# Teste local
if __name__ == "__main__":
    caminho = '/home/daniel/Downloads/WhatsApp-Ptt-2025-04-29-at-09.55.04.mp3'
    transcrever_com_diarizacao(caminho)
