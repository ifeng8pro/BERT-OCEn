import pandas as pd
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
from pytorch_pretrained_bert import BertTokenizer
from fine_tune_bert.finetune_bert import fine_tune_from_datasets

import torch


class Nanchang_Dataset(TorchnlpDataset):

    def __init__(self, root: str, normal_class=0, train_proportion=10,tokenizer='bert', use_tfidf_weights=False, append_sos=False,
                 append_eos=False, clean_txt=False):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        #classes = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'ship']


        classes = ['OfficialReception', 'OfficialVehicles', 'ConventionExpense', 'Remuneration', 'PropertyManagement', 'Procurement', 'NULL']

        self.normal_classes = [classes[normal_class]]
        print('self.normal_classes',self.normal_classes)
        del classes[normal_class]
        self.outlier_classes = classes

        # Load the reuters dataset
        #self.train_set, self.test_set = reuters_dataset(directory=root, train=True, test=True, clean_txt=clean_txt)

        self.train_set, self.test_set = nc_dataset(directory=root, train=True, test=True, clean_txt=clean_txt, train_proportion=train_proportion)
        #print('self.test_set', self.test_set)


        fine_tune_from_datasets(
            train_set=self.train_set,
            normal_classes=self.normal_classes,
            epochs=2,
            batch_size=32,
            max_len=128
        )

        # Pre-process
        self.train_set.columns.add('index')
        self.test_set.columns.add('index')
        self.train_set.columns.add('weight')
        self.test_set.columns.add('weight')

        print('train_set.columns', self.train_set.columns)

        text_for_bert = []  # for finetune
        label_for_bert = []  # for finetune
        train_idx_normal = []  # for subsetting train_set to normal class
        for i, row in enumerate(self.train_set):
            if row['label'] in self.normal_classes:
                train_idx_normal.append(i)
                row['label'] = torch.tensor(0)
                label_for_bert.append(0)  # for finetune
            else:
                row['label'] = torch.tensor(1)
                label_for_bert.append(1)  # for finetune

            row['text'] = row['text'].lower()
            text_for_bert.append(row['text'])  # for finetune




        test_idx = []  # for subsetting test_set to selected normal and anomalous classes
        for i, row in enumerate(self.test_set):
            if row['label'] in self.normal_classes:
                test_idx.append(i)
                row['label'] = torch.tensor(0)
            elif row['label'] in self.outlier_classes:
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
            print('tokenizer == spacy')
            self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=append_eos)
        if tokenizer == 'bert':
            print('tokenizer == bert')
            self.encoder = MyBertTokenizer.from_pretrained('bert-base-chinese', cache_dir=root)

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

def nc_dataset(directory='../data', train=True, test=False, clean_txt=False, train_proportion=10):
    """
    Load and preprocess the nanchang dataset for text classification.

    Args:
        directory: Path to the dataset directory
        train: Whether to load training data
        test: Whether to load test data
        clean_txt: Whether to clean text
        train_proportion: Percentage of data to use for training (0-100)

    Returns:
        Tuple containing (train_dataset, test_dataset)

    """

    ret = []

    # Load the full dataset
    df_data = pd.read_excel(directory + '/corpora/nancahng.xlsx')

    # Check for required columns
    required_columns = ['index', 'Purpose', 'Category']
    
    for col in required_columns:
        if col not in df_data.columns:
            raise ValueError(f"Dataset missing required column: {col}")

    # Initialize storage lists
    train_examples = []
    test_examples = []

    # Group by category and sample training data
    grouped = df_data.groupby('Category')
    for label, group in grouped:
        # Calculate number of training samples for this category
        class_total = len(group)
        # Ensure at least 5 training samples per category
        class_train_size = max(5, int(class_total * (train_proportion / 100)))
        # Randomly sample training indices
        train_indices = group.sample(n=class_train_size, random_state=42).index
        # Split into train and test
        train_group = group.loc[train_indices]

        # Add to training set
        for idx, row in train_group.iterrows():
            train_examples.append({
                'my_index': row['index'],
                'text': str(row['Purpose']),
                'label': row['Category']
            })

    # Get all data for testing
    index_test = df_data['index'].values.tolist()
    label_test = df_data['Category'].values.tolist()
    text_test = df_data['Purpose'].values.tolist()

    for index, text in enumerate(text_test):
        test_examples.append({
            'my_index': index_test[index],
            'text': text,
            'label': label_test[index],
        })

    # Create dataset objects
    ret.append(Dataset(train_examples))
    ret.append(Dataset(test_examples))

    # Print statistics
    total_samples = len(df_data)
    train_count = len(train_examples)
    test_count = len(test_examples)

    print(f'Training set size: {train_count} ({train_count / total_samples * 100:.1f}% of total)')
    print(f'Test set size: {test_count} ({test_count / total_samples * 100:.1f}% of total)')
    print(f'Total samples: {total_samples}')

    return tuple(ret)

