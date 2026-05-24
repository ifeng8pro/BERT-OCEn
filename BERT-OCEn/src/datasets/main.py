from .reuters21578 import Reuters_Dataset
from .newsgroups20 import Newsgroups20_Dataset
from .imdb import IMDB_Dataset
from .yixing import Yixing_Dataset
from .nanchang import Nanchang_Dataset
from .r8 import R8_Dataset


def load_dataset(dataset_name, data_path, normal_class,train_proportion, tokenizer='spacy', use_tfidf_weights=False,
                 append_sos=False, append_eos=False, clean_txt=False):
    """Loads the dataset."""

    implemented_datasets = ('reuters', 'newsgroups20', 'imdb', 'yixing', 'nanchang','r8')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'reuters':
        dataset = Reuters_Dataset(root=data_path, normal_class=normal_class, tokenizer=tokenizer,
                                  use_tfidf_weights=use_tfidf_weights, append_sos=append_sos, append_eos=append_eos,
                                  clean_txt=clean_txt)

    if dataset_name == 'newsgroups20':
        dataset = Newsgroups20_Dataset(root=data_path, normal_class=normal_class, tokenizer=tokenizer,
                                       use_tfidf_weights=use_tfidf_weights, append_sos=append_sos,
                                       append_eos=append_eos, clean_txt=clean_txt)

    if dataset_name == 'imdb':
        dataset = IMDB_Dataset(root=data_path, normal_class=normal_class, tokenizer=tokenizer,
                               use_tfidf_weights=use_tfidf_weights, append_sos=append_sos, append_eos=append_eos,
                               clean_txt=clean_txt)

    if dataset_name == 'yixing':
        dataset = Yixing_Dataset(root=data_path, normal_class=normal_class, train_proportion=train_proportion,
                                  tokenizer=tokenizer,use_tfidf_weights=use_tfidf_weights, append_sos=append_sos,
                                  append_eos=append_eos,clean_txt=clean_txt)

    if dataset_name == 'nanchang':
        dataset = Nanchang_Dataset(root=data_path, normal_class=normal_class, train_proportion=train_proportion,
                                  tokenizer=tokenizer,use_tfidf_weights=use_tfidf_weights, append_sos=append_sos,
                                  append_eos=append_eos,clean_txt=clean_txt)

    if dataset_name == 'r8':
        dataset = R8_Dataset(root=data_path, normal_class=normal_class, tokenizer=tokenizer,
                                  use_tfidf_weights=use_tfidf_weights, append_sos=append_sos, append_eos=append_eos,
                                  clean_txt=clean_txt)

    return dataset
