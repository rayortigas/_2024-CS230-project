from typing import Dict, List

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from lora import LoRALayer, LoRAUpdate



def train_sst2():
    sst2 = load_dataset("stanfordnlp/sst2")

    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
    def tokenize(batch: Dict[str, List]):
        return tokenizer(batch["sentence"], truncation=True, max_length=512)
    
    tokenized_sst2 = sst2.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction
        return accuracy.compute(
            predictions=np.argmax(predictions, axis=1),
            references=labels,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased",
        num_labels=2,
    )

    model = model.to("cuda")

    training_args = TrainingArguments(
        output_dir="tmp/mobilebert",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_sst2["train"],
        eval_dataset=tokenized_sst2["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    torch.save(model.state_dict(), "base-sst2.pt")

if __name__ == "__main__":
    train_sst2()
