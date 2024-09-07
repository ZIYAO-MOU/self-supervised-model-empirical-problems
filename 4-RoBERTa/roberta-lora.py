
"""
Train a RoBERTa model with LoRA on the SST-2 dataset for classification.
"""
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import numpy as np
import evaluate

# Get dataset
ds = load_dataset("stanfordnlp/sst2")

# load tokenizer and model and metric
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
metric = evaluate.load("accuracy")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

encoded_dataset = ds.map(tokenize_function, batched=True)

# Add LoRA to the model
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,   
    inference_mode=False,         
    r=8,                          
    lora_alpha=32,                
    lora_dropout=0.1              
)

model = get_peft_model(model, lora_config)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    num_train_epochs=5,
    logging_dir="logs/lora/",
    report_to="tensorboard",
    logging_strategy="steps",
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    output_dir="checkpoints/lora/",
)

# Split val into val and test
test_size = 400

encoded_dataset["test"] = encoded_dataset["validation"].select(range(test_size))
encoded_dataset["validation"] = encoded_dataset["validation"].select(range(test_size, len(encoded_dataset["validation"])))

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

test_result = trainer.evaluate(encoded_dataset["test"])
val_result =  trainer.evaluate(encoded_dataset["validation"])

print(f'test_result:{test_result}, val_result: {val_result}')

# Save the model
model.save_pretrained("best/lora/")
