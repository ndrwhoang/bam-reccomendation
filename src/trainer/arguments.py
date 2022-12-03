import os
import configparser

from transformers import TrainingArguments


config = configparser.ConfigParser()
config.read(os.path.normpath("configs/base_config.ini"))
config = config["trainer"]

training_args = TrainingArguments(
    run_name=config["run_name"],
    output_dir=os.path.normpath("./training_out/"),
    evaluation_strategy="epoch",
    per_device_train_batch_size=config.geint("train_bs"),
    per_device_eval_batch_size=config.getint("eval_bs"),
    num_train_epochs=config.getint("n_epochs"),
    log_level="info",
    save_strategy="epoch",
    save_total_limit=config.getint("save_total_limit"),
    seed=config.getint("seed"),
    gradient_accumulation_steps=config.getint("gradient_accumulation_steps"),
    metric_for_best_model="eval_accuracy",
    gradient_checkpointing=config.getbool("gradient_checkpointing"),
    fp16=config.getbool("fp16"),
    # deepspeed=None
)
