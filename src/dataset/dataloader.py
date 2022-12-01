import json
import os
from tqdm import tqdm
import random
import logging
from itertools import chain

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

random.seed(123)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseDataset(Dataset):
    def __init__(self, config, tokenizer, mode="train"):
        self.config = config
        self.mode = mode
        self.tokenizer = tokenizer

        data = self._read_data()
        self.queries, self.candidates = self._convert_to_samples(data)
        self.n_samples = len(self.queries)

    def _read_data(self):
        # Read in dataset and mapping for hard candidates
        dataset_path = os.path.normpath(self.config["path"]["dataset"])
        if self.mode == "train":
            indices_path = self.config["path"]["train_indices"]
            hard_mapping_path = self.config["path"]["train_hard_indices"]
        elif self.mode == "train_subset":
            indices_path = self.config["path"]["train_subset_indices"]
            hard_mapping_path = self.config["path"]["train_subset_hard_indices"]
        elif self.mode == "val":
            indices_path = self.config["path"]["val_indices"]
            hard_mapping_path = self.config["path"]["val_hard_indices"]
        elif self.mode == "test":
            raise NotImplementedError
        indices_path = os.path.normpath(indices_path)
        hard_mapping_path = os.path.normpath(hard_mapping_path)

        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        with open(indices_path, "r", encoding="utf-8") as f:
            sample_ids = json.load(f)

        with open(hard_mapping_path, "r", encoding="utf-8") as f:
            hard_mapping = json.load(f)

        dataset = {sample_id: dataset[sample_id] for sample_id in sample_ids}

        return {"dataset": dataset, "hard_mapping": hard_mapping}

    def _convert_to_samples(self, data):
        # Build list of samples and corresponding hard candidates
        dataset = data["dataset"]
        mapping = data["hard_mapping"]

        queries = []
        candidates = []

        for sample_id, sample in tqdm(dataset.items()):
            text = sample["synopsis"]
            hard_candidate_ids = mapping[sample_id]
            random.shuffle(hard_candidate_ids)
            hard_candidate_ids = hard_candidate_ids[
                : int(self.config["dataset"]["n_hard_candidates"])
            ]
            hard_candidates = [
                dataset[cand_id]["synopsis"] for cand_id in hard_candidate_ids
            ]
            hard_candidates.insert(0, text)

            queries.append(text)
            candidates.append(hard_candidates)

        return queries, candidates

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.queries[index], self.candidates[index]

    def collate_fn(self, batch):
        queries, candidates = zip(*batch)
        candidates = list(chain.from_iterable(candidates))

        query_encoding = self.tokenizer(
            list(queries),
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
            max_length=1024,
        )

        candidate_encoding = self.tokenizer(
            candidates,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
            max_length=1024,
        )

        query_bs = query_encoding["input_ids"].size(0)
        candidate_bs = candidate_encoding["input_ids"].size(0)
        n_groups = int(candidate_bs / query_bs)
        labels = torch.arange(0, candidate_bs, n_groups)

        return {
            "query_ids": query_encoding["input_ids"],
            "query_attention_mask": query_encoding["attention_mask"],
            "candidate_ids": candidate_encoding["input_ids"],
            "candidate_attention_mask": candidate_encoding["attention_mask"],
            "labels": labels,
        }
