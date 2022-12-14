import os
import configparser

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.dataset.dataloader import BaseDataset


def test_dataloader(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_name"])
    dataset = BaseDataset(config, tokenizer, "train")
    dataloader = DataLoader(
        dataset, batch_size=5, shuffle=False, collate_fn=dataset.collate_fn
    )

    for i_batch, batch in enumerate(dataloader):
        if i_batch == 8:
            break

        print(batch["query_ids"].size())
        print(batch["query_attention_mask"].size())
        print(torch.isnan(batch["query_ids"]).any())
        print(torch.isnan(batch["query_attention_mask"]).any())
        print(batch["candidate_ids"].size())
        print(batch["candidate_attention_mask"].size())
        print(torch.isnan(batch["candidate_ids"]).any())
        print(torch.isnan(batch["candidate_attention_mask"]).any())
        print(batch["labels"].size())
        print(batch["labels"])
        print(torch.isnan(batch["labels"]).any())

        if (
            torch.isnan(batch["query_ids"]).any()
            or torch.isnan(batch["query_attention_mask"]).any()
            or torch.isnan(batch["candidate_ids"]).any()
            or torch.isnan(batch["candidate_attention_mask"]).any()
            or torch.isnan(batch["labels"]).any()
        ):
            raise ValueError
        print("============================================")


def test_dataloader_decode_sample(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_name"])
    dataset = BaseDataset(config, tokenizer, "train_subset")
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn
    )

    for i_batch, batch in enumerate(dataloader):
        query = tokenizer.batch_decode(batch["query_ids"])
        print(query[0])
        print("+++")
        candidates = tokenizer.batch_decode(batch["candidate_ids"])
        for candidate in candidates:
            print(candidate)
        print("===========-=-=-=-=-=-")
        print("\n")


if __name__ == "__main__":
    print("hello world")

    config = configparser.ConfigParser()
    config.read(os.path.normpath("configs/base_config.ini"))

    test_dataloader(config)
    # test_dataloader_decode_sample(config)
