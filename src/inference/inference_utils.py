import configparser
from os.path import join, normpath

import torch
from transformers import PretrainedConfig, AutoTokenizer

from src.model.encoder import Reranker
# model.load_state_dict(torch.load(os.path.join(), map_location=device))
# model.eval()

def load_model(config):
    ckpt_dir = config['inference']['ckpt_dir']
    map_location = config['inference']['cuda']
    
    model_config = PretrainedConfig.from_pretrained(ckpt_dir)
    model = Reranker(model_config, inference=True)
    model.load_state_dict(torch.load(join(normpath(ckpt_dir), 'pytorch_model.bin'), map_location=map_location))
    
    return model

def load_tokenizer(config):
    # tokenizer = AutoTokenizer.from_pretrained(normpath(config['inference']['tokenizer_dir']))
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-xsmall')
    
    return tokenizer