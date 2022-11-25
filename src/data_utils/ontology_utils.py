import json
import os
from collections import defaultdict


def calculate_genre_frequency(anime_json, book_json, movie_json):
    # Create ontology and count of all genre in 3 datasets
    anime_path = os.path.normpath(anime_json)
    book_path = os.path.normpath(book_json)
    movie_path = os.path.normpath(movie_json)
    out = defaultdict(int)

    with open(anime_path, "r", encoding="utf-8") as f:
        animes = json.load(f)
    with open(book_path, "r", encoding="utf-8") as f:
        books = json.load(f)
    with open(movie_path, "r", encoding="utf-8") as f:
        movies = json.load(f)

    data = animes + books + movies
    for sample in data:
        for gen in sample["genre"]:
            out[gen.strip().lower().replace("-", " ")] += 1

    path_out = os.path.normpath("dataset/misc/genre_frequency.json")
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(out, f, sort_keys=True)

    return out


def normalize_genre_ontology(original_ontology, mapping, normalized_ontology_path_out):
    # Reduce full set of genre to main genres using manually defined mapping
    ontology_path = os.path.normpath(original_ontology)
    mapping = os.path.normpath(mapping)
    out = {}

    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    with open(mapping, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    for genre, genre_to_merge in mapping.items():
        if genre_to_merge not in out:
            out[genre_to_merge] = ontology[genre_to_merge]

        out[genre_to_merge] += ontology[genre]

    for genre, freq in ontology.items():
        if genre not in out and genre not in mapping and freq > 49:
            out[genre] = freq

    with open(normalized_ontology_path_out, "w", encoding="utf-8") as f:
        json.dump(out, f, sort_keys=True)

    return out


def normalize_data_genre(dataset_json, mapping, normalized_ontology, path_out):
    # Remap samples genre to main genres
    dataset_json = os.path.normpath(dataset_json)
    mapping = os.path.normpath(mapping)
    normalized_ontology = os.path.normpath(normalized_ontology)
    path_out = os.path.join(path_out)

    out = []

    with open(mapping, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    with open(normalized_ontology, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    with open(dataset_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for sample in dataset:
        genres = sample["genre"]
        new_genres = []
        for genre in genres:
            if genre.lower() in mapping:
                new_genres.append(mapping[genre.lower()])
            elif genre.lower() in ontology:
                new_genres.append(genre.lower())

        out.append(
            {
                "title": sample["title"],
                "synopsis": sample["synopsis"],
                "genre": new_genres,
            }
        )

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    return out


# def filter_null_samples(dataset_json, path_out):
#     # Filter dataset for samples with empty genre (naturally empty or genre too niche)
