from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer, TrainingArguments


@dataclass
class DistillationTrainingArguments(TrainingArguments):
    temperature: float = 1.0


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False):
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        loss = self.args.temperature**2.0 * (
            self.kl_loss(
                F.log_softmax(student_logits / self.args.temperature, dim=-1),
                F.softmax(teacher_logits / self.args.temperature, dim=-1),
            )
        )
        return (loss, student_outputs) if return_outputs else loss
