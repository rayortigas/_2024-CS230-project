import argparse
import logging
from pathlib import Path

import evaluate
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import lora
from distillation import DistillationTrainer, DistillationTrainingArguments

logger = logging.getLogger(__name__)


def train(
    args: argparse.Namespace,
    task: str,
    num_labels: int,
    dataset,
    train_dataset_key,
    eval_dataset_key,
    tokenize_fn,
) -> None:
    if args.pretrained_id is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_id,
            num_labels=num_labels,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_weights,
            config=AutoConfig.from_pretrained(
                args.pretrained_config,
                num_labels=num_labels,
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id if args.tokenizer_id is not None else args.pretrained_id
    )

    tokenized_data = dataset.map(
        lambda batch: tokenize_fn(tokenizer, batch, truncation=True, max_length=512),
        batched=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction
        return accuracy.compute(
            predictions=np.argmax(predictions, axis=1),
            references=labels,
        )

    def create_training_args(training_args_cls, **kwargs):
        return training_args_cls(
            output_dir=f"./tmp/{task}",
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
            **kwargs,
        )

    def create_trainer(trainer_cls, training_args, **kwargs):
        return trainer_cls(
            model=model,
            args=training_args,
            train_dataset=tokenized_data[train_dataset_key],
            eval_dataset=tokenized_data[eval_dataset_key],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            **kwargs,
        )

    if args.distillation_temperature is None:
        match args.mode:
            case "sft":
                filename = f"{args.base_output_dir}/{task}_{args.tag}_{args.mode}_seed-{args.seed}.pt"
            case "lora":
                filename = f"{args.base_output_dir}/{task}_{args.tag}_{args.mode}-{args.lora_rank}_seed-{args.seed}.pt"

        training_args = create_training_args(TrainingArguments)
        trainer = create_trainer(Trainer, training_args)
    else:
        teacher_stem = Path(args.distillation_teacher_weights).stem
        match args.mode:
            case "sft":
                filename = f"{args.base_output_dir}/{task}_{args.tag}_{args.mode}_teacher-{teacher_stem}_seed-{args.seed}.pt"
            case "lora":
                filename = f"{args.base_output_dir}/{task}_{args.tag}_{args.mode}-{args.lora_rank}_teacher-{teacher_stem}_seed-{args.seed}.pt"

        training_args = create_training_args(
            DistillationTrainingArguments, temperature=args.distillation_temperature
        )
        if args.distillation_teacher_id is not None:
            teacher_model = AutoModelForSequenceClassification.from_pretrained(
                args.distillation_teacher_id,
                num_labels=num_labels,
            )
        else:
            teacher_model = AutoModelForSequenceClassification.from_config(
                config=AutoConfig.from_pretrained(
                    args.distillation_teacher_config,
                    num_labels=num_labels,
                ),
            )
        if args.distillation_teacher_mode == "lora":
            teacher_model = lora.wrap_bert_model_with_lora(
                teacher_model,
                rank=args.distillation_teacher_lora_rank,
                alpha=args.distillation_teacher_lora_rank,
            )
        teacher_model.load_state_dict(
            torch.load(args.distillation_teacher_weights, weights_only=True)
        )
        teacher_model = teacher_model.to("cuda")
        trainer = create_trainer(
            DistillationTrainer, training_args, teacher_model=teacher_model
        )

    logger.info(f"will train model and save to {filename}")

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
        required=False,
        choices=["sft", "lora"],
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
        "--distillation_temperature",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--distillation_teacher_mode",
        type=str,
        required=False,
        choices=["sft", "lora"],
    )
    parser.add_argument(
        "--distillation_teacher_id",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--distillation_teacher_weights",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--distillation_teacher_config",
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
    parser.add_argument(
        "--distillation_teacher_lora_rank",
        type=int,
        required=False,
    )
    args = parser.parse_args()
    if args.mode == "lora":
        if not args.lora_rank:
            parser.error("`lora_rank` is required if using `lora` mode.")
    if args.distillation_teacher_mode == "lora":
        if not args.distillation_teacher_lora_rank:
            parser.error(
                "`distillation_teacher_lora_rank` is required if using distillation teacher `lora` mode."
            )

    return args
