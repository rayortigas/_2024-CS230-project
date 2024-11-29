import argparse
import logging
from typing import Dict, List

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import lora
from utils import set_seed

logger = logging.getLogger(__name__)


def train_sst2(args: argparse.Namespace) -> None:
    match args.mode:
        case "sft":
            filename = f"{args.base_output_dir}/sst2_{args.tag}_{args.mode}_seed-{args.seed}.pt"
        case "lora":
            filename = f"{args.base_output_dir}/sst2_{args.tag}_{args.mode}-{args.lora_rank}_seed-{args.seed}.pt"

    logger.info(f"will train model and save to {filename}")

    if args.pretrained_id is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_id,
            num_labels=2,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_weights,
            config=AutoConfig.from_pretrained(
                args.pretrained_config,
                num_labels=2,
            ),
        )

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

    sst2 = load_dataset("stanfordnlp/sst2")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id if args.tokenizer_id is not None else args.pretrained_id
    )

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

    training_args = TrainingArguments(
        output_dir=f"tmp/sst2-{args.mode}",
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir="./logs",
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

    torch.save(model.state_dict(), filename)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["sft", "lora"],
        default="sft",
    )
    parser.add_argument(
        "--pretrained_id",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--pretrained_config",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--tokenizer_id",
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
        "--train_batch_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=False,
        default=0.01,
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
    set_seed(args.seed)
    train_sst2(args)
