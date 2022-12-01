import os
import configparser

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.dataset.dataloader import BaseDataset


def test_dataloader(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_name"])
    dataset = BaseDataset(config, tokenizer, "train_subset")
    dataloader = DataLoader(
        dataset, batch_size=5, shuffle=False, collate_fn=dataset.collate_fn
    )

    for i_batch, batch in enumerate(dataloader):
        if i_batch == 8:
            break

        print(batch["query_ids"].size())
        print(batch["query_attention_mask"].size())
        print(batch["candidate_ids"].size())
        print(batch["candidate_attention_mask"].size())
        print(batch["labels"].size())
        print(batch["labels"])
        print("============================================")


if __name__ == "__main__":
    print("hello world")

    config = configparser.ConfigParser()
    config.read(os.path.normpath("configs/base_config.ini"))

    test_dataloader(config)
