import configparser
from os.path import join, normpath

import torch
from transformers import AutoTokenizer, AutoConfig

from src.model.encoder import Reranker
# model.load_state_dict(torch.load(os.path.join(), map_location=device))
# model.eval()

def load_model(config):
    ckpt_dir = config['inference']['ckpt_dir']
    map_location = config['inference']['device']
    
    model_config = AutoConfig.from_pretrained(normpath(ckpt_dir))
    # print(model_config)
    model = Reranker(model_config, inference=True)
    model.load_state_dict(torch.load(join(normpath(ckpt_dir), 'pytorch_model.bin'), map_location=map_location))
    
    return model

def load_tokenizer(config):
    # tokenizer = AutoTokenizer.from_pretrained(normpath(config['inference']['tokenizer_dir']))
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-xsmall')
    
    return tokenizer