import os
import logging
import configparser

from torch.utils.data import DataLoader
from transformers import Trainer, PretrainedConfig, AutoTokenizer, EarlyStoppingCallback

from src.dataset.dataloader import BaseDataset
from src.model.encoder import Reranker
from src.model.model_utils import update_pretrained_config
from src.trainer.arguments import get_training_args
from src.trainer.custom_callbacks import freeze_model, UnfreezingCallback
from src.trainer.custom_metrics import compute_metrics
from src.trainer.custom_optimizers import get_adafactor_os


def load_model(config):
    pretrained_name = config['model']['pretrained_name']
    pretrained_config = PretrainedConfig.from_pretrained(pretrained_name)
    pretrained_config = update_pretrained_config(pretrained_config, config['model'])
    model = Reranker(pretrained_config)
    model = freeze_model(model, [0,1,2,3,4])

    return model


def load_dataset(config):
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_name'])
    dataset = BaseDataset(config, tokenizer, 'train_subset')
    
    return dataset


def init_trainer(config):
    training_args = get_training_args(config['trainer'])
    model = load_model(config)  
    optimizer, lr_scheduler = get_adafactor_os(model)
    dataset = load_dataset(config)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=dataset.collate_fn,
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler)
    )
    
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=config.getint('trainer', 'early_stopping')),
        UnfreezingCallback(config.getint('trainer', 'unfreeze_epoch'), trainer)
    ]
    
    for callback in callbacks:
        trainer.add_callback(callback)
    
    return trainer
    

if __name__ == '__main__':
    print('hello world')
    
    config_path = 'configs/base_config.ini'
    
    config = configparser.ConfigParser()
    config.read(os.path.normpath(config_path))
    
    trainer = init_trainer(config)
    # trainer.train()