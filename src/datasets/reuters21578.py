import os
import random
from base.torchnlp_dataset import TorchnlpDataset
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from torch.utils.data import Subset
from nltk.corpus import reuters
from nltk import word_tokenize
from utils.text_encoders import MyBertTokenizer
from utils.misc import clean_text
from .preprocessing import compute_tfidf_weights

import torch
import nltk
import pandas as pd

class Reuters_Dataset(TorchnlpDataset):

    def __init__(self, root: str, normal_class=0, tokenizer='bert', use_tfidf_weights=False, append_sos=False,
                 append_eos=False, clean_txt=False):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        classes = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'ship']

        # classes_full_list = [
        #     'acq', 'alum', 'barley', 'bop', 'carcass', 'castor-oil', 'cocoa', 'coconut', 'coconut-oil', 'coffee',
        #     'copper', 'copra-cake', 'corn', 'cotton', 'cotton-oil', 'cpi', 'cpu', 'crude', 'dfl', 'dlr', 'dmk',
        #     'earn', 'fuel', 'gas', 'gnp', 'gold', 'grain', 'groundnut', 'groundnut-oil', 'heat', 'hog', 'housing',
        #     'income', 'instal-debt', 'interest', 'ipi', 'iron-steel', 'jet', 'jobs', 'l-cattle', 'lead', 'lei',
        #     'lin-oil', 'livestock', 'lumber', 'meal-feed', 'money-fx', 'money-supply', 'naphtha', 'nat-gas', 'nickel',
        #     'nkr', 'nzdlr', 'oat', 'oilseed', 'orange', 'palladium', 'palm-oil', 'palmkernel', 'pet-chem', 'platinum',
        #     'potato', 'propane', 'rand', 'rape-oil', 'rapeseed', 'reserves', 'retail', 'rice', 'rubber', 'rye',
        #     'ship', 'silver', 'sorghum', 'soy-meal', 'soy-oil', 'soybean', 'strategic-metal', 'sugar', 'sun-meal',
        #     'sun-oil', 'sunseed', 'tea', 'tin', 'trade', 'veg-oil', 'wheat', 'wpi', 'yen', 'zinc'
        # ]

        self.normal_classes = [classes[normal_class]]
        del classes[normal_class]
        self.outlier_classes = classes

        # Load the reuters dataset
        self.train_set, self.test_set = reuters_dataset(directory=root, train=True, test=True, clean_txt=clean_txt)

        # Pre-process
        self.train_set.columns.add('index')
        self.test_set.columns.add('index')
        self.train_set.columns.add('weight')
        self.test_set.columns.add('weight')
        self.train_set.columns.add('my_index')
        self.test_set.columns.add('my_index')

        text_for_bert = []  # for finetune
        label_for_bert = []  # for finetune
        train_idx_normal = []  # for subsetting train_set to normal class
        class_labels = []
        for i, row in enumerate(self.train_set):
            if any(label in self.normal_classes for label in row['label']) and (len(row['label']) == 1):
                train_idx_normal.append(i)
                class_labels.append(row['label'])
                row['label'] = torch.tensor(0)
                label_for_bert.append(0)  # for finetune

            else:
                class_labels.append(row['label'])

                row['label'] = torch.tensor(1)
                label_for_bert.append(1)  # for finetune

            row['text'] = row['text'].lower()
            text_for_bert.append(row['text'])  # for finetune
        df_data = pd.DataFrame({'Purpose': text_for_bert, 'label': label_for_bert,'Category':class_labels})



        test_idx = []  # for subsetting test_set to selected normal and anomalous classes
        for i, row in enumerate(self.test_set):
            if any(label in self.normal_classes for label in row['label']) and (len(row['label']) == 1):
                test_idx.append(i)
                row['label'] = torch.tensor(0)
            elif any(label in self.outlier_classes for label in row['label']) and (len(row['label']) == 1):
                test_idx.append(i)
                row['label'] = torch.tensor(1)
            else:
                row['label'] = torch.tensor(1)
            row['text'] = row['text'].lower()

        # Subset train_set to normal class
        self.train_set = Subset(self.train_set, train_idx_normal)
        # Subset test_set to selected normal and anomalous classes
        self.test_set = Subset(self.test_set, test_idx)

        # Make corpus and set encoder
        text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.test_set)]
        if tokenizer == 'spacy':
            self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=append_eos)
        if tokenizer == 'bert':
            self.encoder = MyBertTokenizer.from_pretrained('bert-base-uncased', cache_dir=root)

        # Encode
        for row in datasets_iterator(self.train_set, self.test_set):
            if append_sos:
                sos_id = self.encoder.stoi[DEFAULT_SOS_TOKEN]
                row['text'] = torch.cat((torch.tensor(sos_id).unsqueeze(0), self.encoder.encode(row['text'])))
            else:
                row['text'] = self.encoder.encode(row['text'])

        # Compute tf-idf weights
        if use_tfidf_weights:
            compute_tfidf_weights(self.train_set, self.test_set, vocab_size=self.encoder.vocab_size)
        else:
            for row in datasets_iterator(self.train_set, self.test_set):
                row['weight'] = torch.empty(0)

        # Get indices after pre-processing
        for i, row in enumerate(self.train_set):
            row['index'] = i
        for i, row in enumerate(self.test_set):
            row['index'] = i
        # Get indices after pre-processing
        for i, row in enumerate(self.train_set):
            row['my_index'] = i
        for i, row in enumerate(self.test_set):
            row['my_index'] = i


def reuters_dataset(directory='../data', train=True, test=False, clean_txt=False, train_ratio=0.5, random_seed=42):
    """
    Load the Reuters-21578 dataset with optional custom train/test split ratio (fixed random seed).

    Args:
        directory (str): Directory to cache the dataset.
        train (bool): If True, return training split.
        test (bool): If True, return test split.
        clean_txt (bool): If True, clean text before processing.
        train_ratio (float, optional): If provided, randomly split ALL data into train/test with this ratio.
                                      Overrides default 'train'/'test' file-based split.
        random_seed (int): Random seed for reproducible shuffling (if train_ratio is used).

    Returns:
        Dataset or tuple: Returns requested split(s) (train and/or test), with fields:
                         {'text': str, 'label': list}.
    """
    nltk.download('reuters', download_dir=directory)
    if directory not in nltk.data.path:
        nltk.data.path.append(directory)

    # Load all data
    examples = []
    for fileid in reuters.fileids():
        text = reuters.raw(fileid)
        if clean_txt:
            text = clean_text(text)
        else:
            text = ' '.join(word_tokenize(text))
        labels = reuters.categories(fileid)
        examples.append({'text': text, 'label': labels})  # Only keep 'text' and 'label'

    # Random split by ratio (only when train_ratio is not None)
    if train_ratio is not None:
        random.seed(random_seed)  # Set fixed random seed
        random.shuffle(examples)  # Shuffle in place
        split_idx = int(len(examples) * train_ratio)
        train_examples = examples[:split_idx]
        test_examples = examples[split_idx:]
    else:
        # Default split by original file organization (train/* and test/*)
        train_examples = [e for e in examples if fileid.startswith('train') for fileid in reuters.fileids() if
                          e == examples[reuters.fileids().index(fileid)]]
        test_examples = [e for e in examples if fileid.startswith('test') for fileid in reuters.fileids() if
                         e == examples[reuters.fileids().index(fileid)]]

    # Build return Dataset objects (strictly keep fields {'text': ..., 'label': ...})
    ret = []
    if train:
        ret.append(Dataset(train_examples))
    if test:
        ret.append(Dataset(test_examples))

    # Debug prints (optional, can be removed in production)
    print('train_true_len', len(ret[0]))
    print(ret[0][1])
    print('test_true_len', len(ret[1]))
    print(ret[1][1])

    # Maintain original return logic
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)
