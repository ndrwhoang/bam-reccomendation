import configparser
import os
import json

import torch
from transformers import PretrainedConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from src.model.encoder import Reranker
from src.model.model_utils import update_pretrained_config
from src.dataset.dataloader import BaseDataset


def load_model(config):
    pretrained_name = config["model"]["pretrained_name"]
    pretrained_config = PretrainedConfig.from_pretrained(pretrained_name)
    pretrained_config = update_pretrained_config(pretrained_config, config["model"])
    model = Reranker(pretrained_config)

    return model


def model_input_test(model):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    query_ids = torch.randint(0, 20000, (4, 512), device=device)
    query_attention_mask = (torch.rand((4, 512), device=device) < 0.75).int()
    candidate_ids = torch.randint(0, 20000, (16, 512), device=device)
    candidate_attention_mask = (torch.rand((16, 512), device=device) < 0.75).int()

    query_bs = query_ids.size(0)
    candidate_bs = candidate_ids.size(0)
    n_groups = int(candidate_bs / query_bs)
    labels = torch.arange(0, candidate_bs, n_groups, device=device)

    batch = {
        "query_ids": query_ids,
        "query_attention_mask": query_attention_mask,
        "candidate_ids": candidate_ids,
        "candidate_attention_mask": candidate_attention_mask,
        "labels": labels,
    }

    with torch.autograd.set_detect_anomaly(False):
        model.to(device)
        model_out = model(**batch)

        print(model_out.logits.size())
        print(model_out.hidden_states.size())
        print(model_out.loss)
        model_out.loss.backward()

    return


def model_sample_with_backprop_test(model, config):
    # There is a bug here, sometimes loss.backward() will return nan
    # Only sometimes, if the first .backward() pass works, then it will be fine for the rest
    # All tensor during forward are verified to not have nans
    # All samples are verified to have correct shapes / not have nans
    # Depends on the day, this will happen more often than not
    # Learning rate and gradients clipping don't affect this at all
    # Idk why this happens, hardware problem?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_name"])
    dataset = BaseDataset(config, tokenizer, "train_subset")
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.00001)

    model.train()
    model.zero_grad(set_to_none=True)
    
    with torch.autograd.set_detect_anomaly(True):
        for i_batch, batch in enumerate(dataloader):
            # if i_batch == 3:
            #     break

            batch = {k: v.to(device) for k, v in batch.items()}
            model_out = model(**batch)

            # if torch.isfinite(model_out.loss):
            #     model_out.loss.backward()
            #     optimizer.step()

            model_out.loss.backward()
            print(model_out.loss)
            optimizer.step()
            model.zero_grad(set_to_none=True)
            print("-==---")

    
    return


if __name__ == "__main__":
    print("hello world")

    config = configparser.ConfigParser()
    config.read(os.path.normpath("configs/base_config.ini"))
    model = load_model(config)
    model_input_test(model)
    # model_sample_with_backprop_test(model, config)

    # NativeLayerNormBackward0
    # AddmmBackward0
