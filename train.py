import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("data.csv")

# Ensure labels are integers
df["label"] = df["label"].astype(int)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

# Tokenization Function
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize_function)

# Split Dataset
dataset = dataset.train_test_split(test_size=0.2)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Load Model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# Accuracy Function
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {"accuracy": accuracy}

# Training Configuration
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=1
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train Model
trainer.train()

# Evaluate
results = trainer.evaluate()

print("\nEvaluation Results:")
print(results)

# Save Model
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

print("\nModel Saved Successfully!")