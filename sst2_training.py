import argparse
import logging
from typing import Dict, List

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import lora

logger = logging.getLogger(__name__)


def train_sst2(args: argparse.Namespace) -> None:
    filename = f"sst2-{args.mode}.pt"
    logger.info(f"will train model and save to {filename}")

    sst2 = load_dataset("stanfordnlp/sst2")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_id)

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
        args.pretrained_id,
        num_labels=2,
    )

    if args.pretrained_weights is not None:
        load_model(model, args.pretrained_weights, strict=False)

    match args.mode:
        case "lora":
            lora.wrap_bert_model_with_lora(
                model, rank=args.lora_rank, alpha=args.lora_rank
            )

    num_trainable_parameters = sum(
        [param.numel() for param in model.parameters() if param.requires_grad]
    )
    logger.info(f"# trainable parameters: {num_trainable_parameters}")

    model = model.to("cuda")

    training_args = TrainingArguments(
        output_dir=f"tmp/sst2-{args.mode}",
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
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

    torch.save(model.state_dict(), f"sst2-{args.mode}.pt")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["base", "lora"],
        default="base",
    )
    parser.add_argument(
        "--pretrained_id",
        type=str,
        required=False,
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        required=False,
    )
    args = parser.parse_args()
    if args.mode == "lora":
        if not args.lora_rank:
            parser.error("`lora_rank` is required if using `lora` mode.")

    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    train_sst2(args)
