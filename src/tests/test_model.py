import configparser
import os

import torch
from transformers import PretrainedConfig

from src.model.encoder import EncoderModule
from src.model.model_utils import update_pretrained_config


def load_model(config):
    pretrained_name = config['model']['pretrained_name']
    pretrained_config = PretrainedConfig.from_pretrained(pretrained_name)
    pretrained_config = update_pretrained_config(pretrained_config, config['model'])    
    model = EncoderModule.from_pretrained(pretrained_name, config=pretrained_config)
    
    return model


def model_input_test(model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    query_ids = torch.randint(0, 20000, (4, 512), device=device)
    query_attention_mask = (torch.rand((4, 512), device=device) < 0.75).int()
    candidate_ids = torch.randint(0, 20000, (16, 512), device=device)
    candidate_attention_mask = (torch.rand((16, 512), device=device) < 0.75).int()
    
    query_bs = query_ids.size(0)
    candidate_bs = candidate_ids.size(0)
    n_groups = int(candidate_bs / query_bs)
    labels = torch.arange(0, candidate_bs, n_groups, device=device)
    
    batch = {
        'query_ids': query_ids,
        'query_attention_mask': query_attention_mask,
        'candidate_ids': candidate_ids,
        'candidate_attention_mask': candidate_attention_mask,
        'labels': labels
    }
    
    model.to(device)
    model_out = model(**batch)
    
    print(model_out.logits.size())
    print(model_out.loss)
    model_out.loss.backward()    

    return

if __name__ =='__main__':
    print('hello world')
    
    config = configparser.ConfigParser()
    config.read(os.path.normpath('configs/base_config.ini'))
    model = load_model(config)
    model_input_test(model)