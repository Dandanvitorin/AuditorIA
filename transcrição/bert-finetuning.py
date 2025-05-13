import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

import sys
import transformers
import os

print("="*60)
print("🔍 Diagnóstico de Ambiente:")
print(f"Python path........: {sys.executable}")
print(f"Transformers v.....: {transformers.__version__}")
print(f"Conda env (se ativo): {os.environ.get('CONDA_DEFAULT_ENV', 'Não detectado')}")
print("="*60)

# === 1. Carregar dataset local ===
df = pd.read_csv('/home/daniel/Downloads/goemotions_3_pt.csv')

# === 2. Mapear emoções para português ===
emotions_map = {
    "admiration": "admiração", "amusement": "diversão", "anger": "raiva", "annoyance": "aborrecimento",
    "approval": "aprovação", "caring": "carinho", "confusion": "confusão", "curiosity": "curiosidade",
    "desire": "desejo", "disappointment": "decepção", "disapproval": "desaprovação", "disgust": "nojo",
    "embarrassment": "constrangimento", "excitement": "empolgação", "fear": "medo", "gratitude": "gratidão",
    "grief": "luto", "joy": "alegria", "love": "amor", "nervousness": "nervosismo", "optimism": "otimismo",
    "pride": "orgulho", "realization": "percepção", "relief": "alívio", "remorse": "remorso",
    "sadness": "tristeza", "surprise": "surpresa", "neutral": "neutro"
}

def extrair_emocao(row):
    for eng_emotion, pt_emotion in emotions_map.items():
        if row.get(eng_emotion, 0) == 1:
            return pt_emotion
    return "sem_emocao"

df["label"] = df.apply(extrair_emocao, axis=1)
df_final = df[["texto", "label"]].dropna()

# === 3. Dividir em treino e validação ===
train_df, valid_df = train_test_split(df_final, test_size=0.2, stratify=df_final["label"], random_state=42)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
valid_dataset = Dataset.from_pandas(valid_df.reset_index(drop=True))

# === 4. Tokenizar ===
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

def tokenize_batch(batch):
    return tokenizer(batch["texto"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_batch, batched=True)
valid_dataset = valid_dataset.map(tokenize_batch, batched=True)

# === 5. Codificar rótulos ===
labels = sorted(df_final["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

def encode_labels(example):
    example["labels"] = int(label2id[example["label"]])
    return example


train_dataset = train_dataset.map(encode_labels).remove_columns(["label"])
valid_dataset = valid_dataset.map(encode_labels).remove_columns(["label"])


# === 6. Carregar modelo ===
model = AutoModelForSequenceClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# === 7. Métricas ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# === 8. Argumentos de treinamento ===
training_args = TrainingArguments(
    output_dir="../emotion-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# === 9. Criar Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
print(train_dataset.features)

# === 10. Treinar ===
trainer.train()
trainer.save_model("./emotion-model/final")
tokenizer.save_pretrained("./emotion-model/final")