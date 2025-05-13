# arquivo: classificador_emocao.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import gc

# === Carregar modelo treinado uma vez ===
model_path = "/home/daniel/PycharmProjects/classificador_chamadas/emotion-model/final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

clf = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, truncation=True)

def classificar_emocao_frase(frases):
    """
    Recebe uma lista de frases e retorna os top 3 sentimentos por frase.
    """
    resultados_completos = []

    for frase in frases:
        analise = clf(frase)[0]
        analise_ordenada = sorted(analise, key=lambda x: x["score"], reverse=True)
        top_emocoes = analise_ordenada[:3]
        resultados_completos.append({
            "frase": frase,
            "emocao_1": top_emocoes[0]["label"],
            "score_1": top_emocoes[0]["score"],
            "emocao_2": top_emocoes[1]["label"],
            "score_2": top_emocoes[1]["score"],
            "emocao_3": top_emocoes[2]["label"],
            "score_3": top_emocoes[2]["score"],
        })

    return resultados_completos

def classificar_emocao(frases):
    """
    Recebe uma lista de frases e retorna a emoção dominante da conversa inteira.
    """
    texto_completo = " ".join(frases)
    if len(texto_completo) > 1024:
        texto_completo = texto_completo[:1024]  # truncar a string
    analise = clf(texto_completo)[0]
    analise_ordenada = sorted(analise, key=lambda x: x["score"], reverse=True)

    torch.cuda.empty_cache()
    gc.collect()

    return {
        "texto": texto_completo,
        "emocao_1": analise_ordenada[0]["label"],
        "score_1": analise_ordenada[0]["score"],
        "emocao_2": analise_ordenada[1]["label"],
        "score_2": analise_ordenada[1]["score"],
        "emocao_3": analise_ordenada[2]["label"],
        "score_3": analise_ordenada[2]["score"],
    }
