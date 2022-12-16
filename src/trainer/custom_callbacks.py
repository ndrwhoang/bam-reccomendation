import numpy as np
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)


# class FreezingCallback(TrainerCallback):
#     def __init__(self, freeze_layers, n_epoch_thresh, trainer):
#         self.trainer = trainer
#         self.n_epoch_thresh = n_epoch_thresh
#         self.freeze_layers = freeze_layers

#     def on_epoch_begin(self, args:TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#         if state.epoch < self.n_epoch_thresh and self.trainer.model.should_freeze:
#             for layer in self.freeze_layers:
#                 for param in self.trainer.model.encoder.encoder.layer[layer].parameters():
#                     param.require_grad = False
#             for param in self.trainer.model.encoder.embeddings.parameters():
#                 param.require_grad = False

#     def on_save(self, args:TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#         for name, param in self.trainer.model.named_parameters():
#             param.requires_grad = True


def freeze_model(model, freeze_layers):
    # Need to change the
    for layer in freeze_layers:
        for param in model.pretrained_encoder.encoder.layer[layer].parameters():
            param.require_grad = False
    for param in model.pretrained_encoder.embeddings.parameters():
        param.requires_grad = False

    return model


class UnfreezingCallback(TrainerCallback):
    def __init__(self, epoch_threshold, unfreeze_lr, trainer: Trainer):
        self.trainer = trainer
        self.unfreeze_lr = unfreeze_lr
        self.epoch_threshold = epoch_threshold

    def check_metric_value(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metric_value,
        **kwargs
    ):
        if state.best_metric is None or np.greater(metric_value, state.best_metric):
            return False
        return True

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs
    ):
        metric_value = metrics.get(args.metric_for_best_model)
        if (
            self.check_metric_value(args, state, control, metric_value)
            or state.epoch > self.epoch_threshold
        ):
            for name, param in self.trainer.model.named_parameters():
                param.requires_grad = True
            for g in self.trainer.optimizer.param_groups:
                g["lr"] = self.unfreeze_lr

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        # Ensures that the model is always completely unfrozen before saving it to avoid unexpected behaviour.
        for name, param in self.trainer.model.named_parameters():
            param.requires_grad = True


class InferenceEvaluationCallback(TrainerCallback):
    def __init__(self):
        raise NotImplementedError
