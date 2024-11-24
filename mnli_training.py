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


def train_mnli(args: argparse.Namespace) -> None:
    filename = f"mnli-{args.mode}.pt"
    logger.info(f"will train model and save to {filename}")

    mnli = load_dataset("nyu-mll/multi_nli").select_columns(
        ["premise", "hypothesis", "label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_id)

    def tokenize(batch: Dict[str, List]):
        return tokenizer(
            text=batch["premise"],
            text_pair=batch["hypothesis"],
            truncation=True,
            max_length=512,
        )

    tokenized_mnli = mnli.map(tokenize, batched=True)

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
        num_labels=3,
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
        output_dir=f"tmp/mnli-{args.mode}",
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
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
        train_dataset=tokenized_mnli["train"],
        eval_dataset=tokenized_mnli["validation_matched"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    torch.save(model.state_dict(), f"mnli-{args.mode}.pt")


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
    train_mnli(args)
