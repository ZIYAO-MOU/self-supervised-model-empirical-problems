from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import matplotlib.pyplot as plt
import torch

ds = load_dataset("stanfordnlp/sst2")

# load tokenizer and model and metric
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
metric = evaluate.load("accuracy")

def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding="max_length")

encoded_dataset = ds.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.from_numpy(logits), axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Split dataset into validation, and test
test_size = 400
eval_dataset = encoded_dataset["validation"].select(range(test_size))
test_dataset = encoded_dataset["validation"].select(range(test_size, len(encoded_dataset["validation"])))

training_args = TrainingArguments(
    per_device_train_batch_size=16,
    num_train_epochs=5,
    report_to="tensorboard",
    logging_steps=100,
    save_steps=1000,
    save_total_limit=1,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    metric_for_best_model="eval_accuracy",
    output_dir="checkpoints/roberta/",
    logging_dir="logs/roberta/",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate(encoded_dataset["test"])

print(f"Validation Accuracy: {results['eval_accuracy']}")
print(f"Test Accuracy: {results['eval_accuracy']}")

model.save_pretrained("best/roberta/")