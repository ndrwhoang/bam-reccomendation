import os
import configparser

from transformers import TrainingArguments


def get_training_args(config):
    training_args = TrainingArguments(
        run_name=config["run_name"],
        output_dir=os.path.normpath("./training_out/"),
        evaluation_strategy="epoch",
        per_device_train_batch_size=config.getint("train_bs"),
        per_device_eval_batch_size=config.getint("eval_bs"),
        num_train_epochs=config.getint("n_epochs"),
        log_level="info",
        save_strategy="epoch",
        save_total_limit=config.getint("save_total_limit"),
        seed=config.getint("seed"),
        gradient_accumulation_steps=config.getint("gradient_accumulation_steps"),
        metric_for_best_model="eval_accuracy",
        gradient_checkpointing=config.getboolean("gradient_checkpointing"),
        fp16=config.getboolean("fp16"),
        # deepspeed=None,
        load_best_model_at_end=True
    )
    
    return training_args
