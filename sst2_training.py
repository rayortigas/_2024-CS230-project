import argparse
import logging
from typing import Dict, List

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

import training
from utils import set_seed

logger = logging.getLogger(__name__)


def train_sst2(args: argparse.Namespace) -> None:
    sst2 = load_dataset("stanfordnlp/sst2")

    def tokenize(tokenizer: PreTrainedTokenizerBase, batch: Dict[str, List], **kwargs):
        return tokenizer(batch["sentence"], **kwargs)

    training.train(
        args,
        task="sst2",
        num_labels=2,
        dataset=sst2,
        train_dataset_key="train",
        eval_dataset_key="validation",
        tokenize_fn=tokenize,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = training.get_args()
    set_seed(args.seed)
    train_sst2(args)
