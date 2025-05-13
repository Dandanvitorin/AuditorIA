
import re
from collections import Counter
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import torch
import gc

def contar_palavras(transcricoes, top_n=25, modo="completo"):
    """
    Conta palavras mais comuns ignorando identificadores como [SPEAKER_00]
    - modo="completo": retorna (wordcloud_figure, dataframe)
    - modo="simples": retorna lista de tuplas
    """
    todas_as_palavras = []
    for frase in transcricoes:
        # Remove [SPEAKER_xx] do início da frase
        frase_limpa = re.sub(r"\[SPEAKER_\d+\]", "", frase)
        palavras = re.findall(r"\b\w+\b", frase_limpa.lower())
        todas_as_palavras.extend(palavras)

    contagem = Counter(todas_as_palavras)
    mais_comuns = contagem.most_common(top_n)

    if modo == "simples":
        return mais_comuns

    tabela = pd.DataFrame(mais_comuns, columns=["Palavra", "Frequência"])

    wordcloud = WordCloud(width=800, height=400, background_color='white')
    wordcloud.generate_from_frequencies(dict(mais_comuns))
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")

    return fig, tabela

# === 2. Frases proibidas ===
frases_proibidas = [
    "isso não é comigo",
    "não posso resolver",
    "tem que entender que é assim",
    "obrigação sua",
    "não entendeu"
]

def detectar_proibidas(transcricoes, frases_proibidas):
    resultados = []
    for frase in transcricoes:
        for proibida in frases_proibidas:
            if proibida in frase.lower():
                resultados.append((frase, proibida))
    return resultados