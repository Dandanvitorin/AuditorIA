
from transcrição.transcriptor import transcrever_com_diarizacao
from emocao.classificador import classificar_emocao
from palavras.words_classifier import contar_palavras, detectar_proibidas

frases_proibidas = [
    "isso não é comigo",
    "não posso resolver",
    "tem que entender que é assim",
    "obrigação sua",
    "não entendeu"
]


def analisar_audio(audio_file):
    resultado = {}

    # Transcrição com diarização
    transcricao = transcrever_com_diarizacao(audio_file)
    resultado["transcricao"] = transcricao


    # Classificação de emoções
    resultado["emocao"] = classificar_emocao(transcricao)

    # Palavras mais comuns
    wordcloud_fig, tabela_palavras = contar_palavras(transcricao)
    resultado["wordcloud_fig"] = wordcloud_fig
    resultado["tabela_palavras"] = tabela_palavras

    # Frases proibidas
    resultado["proibidas"] = detectar_proibidas(transcricao, frases_proibidas)

    return resultado