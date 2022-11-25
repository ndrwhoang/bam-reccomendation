import pandas as pd
import json
import os
import ast
from tqdm import tqdm
import random
import copy

import scipy
import numpy as np
from sklearn.preprocessing import normalize

random.seed(123)


def anime_to_json(path, path_out):
    # Convert raw anime dataset to json
    path = os.path.normpath(path)
    path_out = os.path.normpath(path_out)
    df = pd.read_csv(path, encoding="utf8")
    df = df[["title", "synopsis", "genre"]]
    out = df.to_dict("records")

    for sample in out:
        sample["synopsis"] = str(sample["synopsis"]).replace("\n", "").replace("\r", "")
        sample["genre"] = ast.literal_eval(sample["genre"])

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    return out


def book_to_json(path, path_out):
    # Convert raw book dataset to json
    path = os.path.normpath(path)
    path_out = os.path.normpath(path_out)
    out = []

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            _, _, title, _, _, genre, synopsis = line.split("\t")
            if len(genre) > 0:
                genre = list(json.loads(genre).values())
            else:
                continue
            out.append({"title": title, "synopsis": synopsis, "genre": genre})

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    return out


def movie_to_json(path, path_out):
    # Convert raw movie dataset to json
    path = os.path.normpath(path)
    path_out = os.path.normpath(path_out)
    df = pd.read_csv(path, encoding="utf8")
    df = df[["title", "plot_synopsis", "tags"]]
    out = df.to_dict("records")

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def filter_useless_samples(dataset_json, path_out):
    # filter out samples with too short synopsis and genre list
    dataset_json = os.path.normpath(dataset_json)
    path_out = os.path.normpath(path_out)

    with open(dataset_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    out = [
        sample
        for sample in dataset
        if len(sample["synopsis"]) > 50 and len(sample["genre"]) > 1
    ]

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    return out


def merge_datasets(all_dir):
    # Join all the json/dataset, give them ids to make splitting train test easier
    all_dir = os.path.normpath(all_dir)
    datasets = ["animes", "movies", "books"]
    out = {}

    for dataset in tqdm(datasets):
        dataset_path = os.path.join(all_dir, f"{dataset}_normalized_filtered.json")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code_initial = dataset[0].upper()
        for i_sample, sample in enumerate(data):
            sample_id = f"{code_initial}{str(i_sample).zfill(6)}"
            out[sample_id] = {
                "title": sample["title"],
                "synopsis": sample["synopsis"],
                "genre": sample["genre"],
            }

    with open(
        os.path.join(all_dir, "all_normalized_filtered.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(out, f, ensure_ascii=False)

    return out


def split_train_test_val(all_path, train_size):
    # Split the dataset to train, val, test, and a small subset for model sanity check purposes
    all_path = os.path.normpath(all_path)

    with open(all_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_ids = list(data.keys())
    random.shuffle(all_ids)

    train_partition = int(train_size * len(all_ids))
    val_test_ids = all_ids[train_partition:]
    train_ids = all_ids[:train_partition]
    train_subset_ids = train_ids[:200]
    val_ids = val_test_ids[: len(val_test_ids) // 2]
    test_ids = val_test_ids[len(val_test_ids) // 2 :]

    train_out = os.path.normpath("dataset/processed/train/all_train.json")
    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_ids, f, ensure_ascii=False)

    val_out = os.path.normpath("dataset/processed/val/all_val.json")
    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_ids, f, ensure_ascii=False)

    test_out = os.path.normpath("dataset/processed/test/all_test.json")
    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(test_ids, f, ensure_ascii=False)

    train_subset_out = os.path.normpath("dataset/processed/train/all_train_subset.json")
    with open(train_subset_out, "w", encoding="utf-8") as f:
        json.dump(train_subset_ids, f, ensure_ascii=False)

    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "train_subset_ids": train_subset_ids,
    }


def load_glove_embedding(embedding_path):
    out = {}
    embedding_path = os.path.normpath(embedding_path)

    with open(embedding_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            out[word] = vector

    return out


def calculate_embedding(dataset, word_embedding, norm=True):
    # Run glove mebdding through the dataset
    out = {}
    null_vec = np.zeros(list(word_embedding.values())[0].shape)

    for sample_id, sample in dataset.items():
        sample_embedding = copy.deepcopy(null_vec)
        for word in sample["synopsis"].split():
            sample_embedding += word_embedding.get(word.lower(), null_vec)
        if norm:
            sample_embedding = normalize(sample_embedding.reshape(1, -1))

        out[sample_id] = sample_embedding

    return out


def find_top_n_closest(embedding_dataset, n_hard_samples):
    # Calculate embedding distance and builds mapping of sample to list of hard samples
    embedding_ids = list(embedding_dataset.keys())
    embedding_matrix = np.concatenate(list(embedding_dataset.values()), axis=0)
    out = {}

    distances = scipy.spatial.distance.cdist(
        embedding_matrix, embedding_matrix, "cosine"
    )
    indices = np.argpartition(distances, n_hard_samples, axis=1)[
        :, :n_hard_samples + 1
    ].tolist()

    for embedding_id, indice in zip(embedding_ids, indices):
        hard_samples = [embedding_ids[i] for i in indice]
        out[embedding_id] = [sample for sample in hard_samples if sample != embedding_id]

    return out


def mine_hard_samples(
    dataset_path, sample_ids_path, embedding_path, n_hard_samples, mapping_out
):
    # Use glove embedding to find closest samples as hard samples
    dataset_path = os.path.normpath(dataset_path)
    sample_ids_path = os.path.normpath(sample_ids_path)
    embedding_path = os.path.normpath(embedding_path)
    mapping_out = os.path.normpath(mapping_out)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(sample_ids_path, "r", encoding="utf-8") as f:
        sample_ids = json.load(f)

    dataset = {sample_id: dataset[sample_id] for sample_id in sample_ids}
    word_embedding = load_glove_embedding(embedding_path)
    embedding_dataset = calculate_embedding(dataset, word_embedding)
    hard_sample_mapping = find_top_n_closest(embedding_dataset, n_hard_samples)

    with open(mapping_out, "w", encoding="utf-8") as f:
        json.dump(hard_sample_mapping, f)

    return hard_sample_mapping
