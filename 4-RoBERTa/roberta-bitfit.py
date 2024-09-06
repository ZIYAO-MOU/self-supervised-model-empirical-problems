from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate

# Get dataset
ds = load_dataset("stanfordnlp/sst2")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
metric = evaluate.load("accuracy")

for name, param in model.named_parameters():
    if "classifier" not in name and "bias" not in name:
        param.requires_grad = False

for name, param in model.named_parameters():
    status = "Trainable" if param.requires_grad else "Frozen"
    print(f"{name}: {status}")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

encoded_dataset = ds.map(tokenize_function, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

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
    output_dir="checkpoints/bitfit/",
    logging_dir="logs/bitfit/",
)

test_size = 400

encoded_dataset["test"] = encoded_dataset["validation"].select(range(test_size))
encoded_dataset["validation"] = encoded_dataset["validation"].select(range(test_size, len(encoded_dataset["validation"])))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate(encoded_dataset["test"])
print(results)

print(f"Validation Accuracy: {results['eval_accuracy']}")
print(f"Test Accuracy: {results['eval_accuracy']}")

# Save the model
model.save_pretrained("best/bitfit/")
