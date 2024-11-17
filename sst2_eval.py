from typing import Dict, List

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

import lora


def collate_sst2_batch(tokenizer: Tokenizer, batch):
    sentences = [example["sentence"] for example in batch]
    encoding = tokenizer(
        text=sentences,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    return {
        "sentence": sentences,
        "label": torch.LongTensor([example["label"] for example in batch]),
        "input_ids": torch.LongTensor(encoding["input_ids"]),
        "token_type_ids": torch.LongTensor(encoding["token_type_ids"]),
        "attention_mask": torch.LongTensor(encoding["attention_mask"]),
    }


def aggregate_mean_cosine_similarities(model0, model1, data_loader):
    def collect_attentions(model, batch):
        outputs = model(
            input_ids=batch["input_ids"].to(model.device),
            token_type_ids=batch["token_type_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
        )
        attentions = np.stack([a.detach().cpu().numpy() for a in outputs.attentions])
        return attentions

    def reshape_attentions(attentions):
        return np.transpose(
            np.reshape(attentions, attentions.shape[:-2] + (-1,)), axes=(1, 0, 2, 3)
        )

    cosine_similarities_batches = []
    for batch in tqdm(data_loader):
        cosine_similarities_batch = (
            nn.functional.cosine_similarity(
                torch.tensor(reshape_attentions(collect_attentions(model0, batch))),
                torch.tensor(reshape_attentions(collect_attentions(model1, batch))),
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        cosine_similarities_batches.append(cosine_similarities_batch)
    cosine_similarities = np.concat(cosine_similarities_batches)
    return np.mean(cosine_similarities, axis=0)


def load_sst2_pt_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased",
        num_labels=2,
        output_attentions=True,
    )
    model = model.to("cuda")
    return model


def load_sst2_ft_model(file, mode, **kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased",
        num_labels=2,
        output_attentions=True,
    )

    match mode:
        case "lora":
            lora.wrap_bert_model_with_lora(
                model, "mobilebert", rank=kwargs["lora_rank"], alpha=kwargs["lora_rank"]
            )
    model.load_state_dict(torch.load(file, weights_only=True))
    model = model.to("cuda")
    return model
