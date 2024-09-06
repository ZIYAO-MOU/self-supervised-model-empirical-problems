from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import PeftModel, PeftConfig
import numpy as np
import evaluate
import torch


# Get dataset
ds = load_dataset("stanfordnlp/sst2")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base",num_labels=2, clean_up_tokenization_spaces=True)

base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# model_path = "scripts/best/lora/"
model_path = "/home/cs601-zmou1/4-RoBERTa/scripts/checkpoints/roberta/checkpoint-20000"
model = RobertaForSequenceClassification.from_pretrained(model_path)
# model = PeftModel.from_pretrained(base_model, model_path)

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
print("Val results: ", val_results)