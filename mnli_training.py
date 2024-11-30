import argparse
import logging
from typing import Dict, List

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

import training
from utils import set_seed

logger = logging.getLogger(__name__)


def train_mnli(args: argparse.Namespace) -> None:
    mnli = load_dataset("nyu-mll/multi_nli").select_columns(
        ["premise", "hypothesis", "label"]
    )

    def tokenize(tokenizer: PreTrainedTokenizerBase, batch: Dict[str, List], **kwargs):
        return tokenizer(
            text=batch["premise"],
            text_pair=batch["hypothesis"],
            **kwargs,
        )

    training.train(
        args,
        task="mnli",
        num_labels=3,
        dataset=mnli,
        train_dataset_key="train",
        eval_dataset_key="validation_matched",
        tokenize_fn=tokenize,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = training.get_args()
    set_seed(args.seed)
    train_mnli(args)
