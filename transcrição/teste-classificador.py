from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# === 1. Carregar modelo treinado ===
#model_path = "../emotion-model/final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# === 2. Criar pipeline de classificação ===
clf = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# === 3. Simular interações de operadores com clientes ===
interacoes = [
    "Olá, tudo bem? Estou te ligando para falar sobre uma oportunidade exclusiva.",
    "Senhor, se não fechar hoje, você vai perder a promoção.",
    "Agradeço muito pela sua atenção, qualquer coisa estou à disposição.",
    "Não posso te ajudar com isso, tente outro canal, por favor.",
    "Fique tranquilo, vou resolver isso pra você agora mesmo!",
    "O produto já foi enviado, mas não é minha responsabilidade se houver atraso.",
    "Você gostaria de receber por e-mail ou WhatsApp?",
    "Infelizmente não posso fazer nada.",
    "[SPEAKER_00] Bom, vou te enviar um áudio porque eu acho mais fácil e eu também acho mais confortável.",
    "[SPEAKER_00] Obrigado.",
    "[SPEAKER_00] Eu não quis transmitir como um ultimato. Eu quis transmitir como um transmissor. As condições são tipo, gente, a gente tem que acertar.",
    "[SPEAKER_00] as coisas, mas não tipo assim, mano, se você errar, acabou, tá ligado? Até porque nada vai ser só de certo, tipo, não existe isso, né? Não é o que tava na minha cabeça. Acho que o que tava na minha cabeça são as insistências em erros, tipo, pô, se a gente começar a afastar de novo. Aí sim, tipo assim, mas constância, sabe? Tipo assim, não é sobre ir do zero ao 100, mas tipo assim, sobre as escolhas de adas que a gente vai fazer de estar perto, sabe? Então não era para ser um timato, tá bom? Era para ser realmente, tipo assim, vamos saber se é um norte para a gente alcançar e caminhar em direção a ele juntos. Se não for caminhar em direção a ele, aí a gente repensa as rotas, entendeu? Mas não um timato, é que eu quis te dar uma resposta clara para não te deixar uma resposta vaga, né? Que eu não queria te deixar ansioso, então talvez não tenha me expressado da melhor forma, mas foi mais ou menos isso que eu pensei."
]

# === 4. Rodar predições ===
for frase in interacoes:
    print(f"\n🗣️ Frase: {frase}")
    resultados = clf(frase)[0]
    resultados = sorted(resultados, key=lambda x: x["score"], reverse=True)
    for item in resultados[:3]:  # Mostra top 3 emoções
        print(f" - {item['label']}: {item['score']:.2%}")
