"""
Train a RoBERTa model on the SST-2 dataset for classification.
"""
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from datasets import load_dataset
import evaluate
import torch
from peft import PeftModel, PeftConfig

# Get dataset
ds = load_dataset("stanfordnlp/sst2")

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Load the model from checkpoint
# model = RobertaForSequenceClassification.from_pretrained("/export/b08/nbafna1/projects/courses/601.771-ssl/601.771/checkpoints/roberta_lora/checkpoint-22000")
# base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model_path = "scripts/best/bitfit/"
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_ds = ds.map(tokenize_function, batched=True)

accuracy_metric = evaluate.load("accuracy")

model.eval()

# Calculate accuracy of the model on the test set

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

evaluator = Trainer(model=model, compute_metrics=compute_metrics)

# Take subset of the validation set
test_size = 400
tokenized_ds["test"] = tokenized_ds["validation"].select(range(test_size))
tokenized_ds["validation"] = tokenized_ds["validation"].select(range(test_size, len(tokenized_ds["validation"])))

test_results = evaluator.evaluate(tokenized_ds["test"])
val_results = evaluator.evaluate(tokenized_ds["validation"])

print("Test results: ", test_results)
print("Validation results: ", val_results)