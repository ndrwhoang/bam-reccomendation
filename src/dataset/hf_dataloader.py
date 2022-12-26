import json
from os.path import join, normpath
import glob

import datasets


_DESCRIPTION = """
Dataset of combined anime, book, and movies reviews
"""

_CITATION = """
"""

_DATASET_ROOT = normpath('dataset/processed')
_DATASET_SPLITS = {
    'data_source': join(_DATASET_ROOT, 'all', 'all_normalized_filtered.json'),
    'train_subset': join(_DATASET_ROOT, 'train', 'all_train_subset.json'),
    'train_subset_hards': join(_DATASET_ROOT, 'train', 'all_train_subset_hards.json'),
    'train': join(_DATASET_ROOT, 'train', 'all_train.json'),
    'train_hards': join(_DATASET_ROOT, 'train', 'all_train_hards.json'),
    'val': join(_DATASET_ROOT, 'val', 'all_val.json'),
    'val_hards': join(_DATASET_ROOT, 'val', 'all_val_hards.json'),
    'test': join(_DATASET_ROOT, 'test', 'all_test.json'),
    'test_hards': join(_DATASET_ROOT, 'test', 'all_test_hards.json')
}


class BAMReviewsConfig(datasets.BuilderConfig):
    """ BuilderConfig for BAM review """
    
    def __init__(self, **kwargs):
        """BuilderConfig for BAM review
        
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(BAMReviewsConfig, self).__init__(**kwargs)


class HfDataloader(datasets.GeneratorBasedBuilder):
    """BAM reviews dataset"""
    
    BUILDER_CONFIGS = [
        BAMReviewsConfig(
            name='plain_text',
            version=datasets.Version('1.0.0', ''),
            description='Plain text'
        )
    ]
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'query': datasets.Value('string'),
                    'candidates': [datasets.Value('string')]
                }
            ),
            homepage='https://www.kaggle.com/',
            citation=_CITATION
        )
    
    def _split_generators(self, dl_manager):
        datasets = dl_manager.download_and_extract(_DATASET_SPLITS)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'dataset_path': datasets['data_source'],
                    'index_path': datasets['train'],
                    'index_hards_path': datasets['train_hards']
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    'dataset_path': datasets['data_source'],
                    'index_path': datasets['val'],
                    'index_hards_path': datasets['val_hards']
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'dataset_path': datasets['data_source'],
                    'index_path': datasets['test'],
                    'index_hards_path': datasets['test_hards']
                }
            ),
            datasets.SplitGenerator(
                name='train_subset',
                gen_kwargs={
                    'dataset_path': datasets['data_source'],
                    'index_path': datasets['train_subset'],
                    'index_hards_path': datasets['train_subset_hards']
                }
            )
        ]
    
    def _generate_examples(self, dataset_path, index_path, index_hards_path):
        with open(dataset_path, 'r', encoding='utf-8') as df:
            dataset = json.load(df)
        
        with open(index_hards_path, 'r', encoding='utf-8') as hf:
            hards = json.load(hf)
        
        key = 0
        with open(index_path, 'r', encoding='utf-8') as f:
            sample_indices = json.load(f)
            
            for indice in sample_indices:
                query = dataset[indice]['synopsis']
                hard_indices = hards[indice]
                candidates = [
                    dataset[h_indice]['synopsis'] for h_indice in hard_indices
                ]
                
                yield key, {
                    'query': query,
                    'candidates': candidates
                }
                key += 1
    