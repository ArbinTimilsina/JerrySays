import csv
import os
from typing import List

import spacy
from nltk import sent_tokenize
from torchtext.data import Dataset, Example, Field, Iterator, batch

SPACY_EN = spacy.load('en')

THIS_DIR = os.path.dirname(__file__)


def tokenize_en(text: str) -> List[str]:
    """
    Tokenize English text from a string into a list of strings
    """
    return [tok.text for tok in SPACY_EN.tokenizer(text)]


def get_sentences_by_jerry() -> List[str]:
    """
    Get all the dialogues by Jerry in the CSV file, split each dialogues into
    sentences, and return a list containing the sentences.
    """
    seinfeld_scripts = os.path.join(
        THIS_DIR, '..', 'seinfeld_scripts', 'complete_seinfeld_scripts.csv'
    )

    sentences_by_jerry = []
    with open(seinfeld_scripts) as input_file:
        input_data = csv.DictReader(input_file)
        for row in input_data:
            if row['Character'] == 'JERRY':
                dialogue = row['Dialogue']
                for sentence in sent_tokenize(dialogue):
                    sentences_by_jerry.append(sentence)

    return sentences_by_jerry


def build_dataset_and_vocab(sentences: List[str]):
    """
    Define source and target fields, iterate over the list of sentences to
    create list of Examples, and return:
        - training and validation dataset (split 90-10%)
        - source and target fields with Vocab object
    """
    # Minimum length for sentences to be included in the dataset
    min_length = 6

    # Define source and target fields
    bos_word = '<s>'
    eos_word = '</s>'
    pad_word = '<pad>'
    src_field = Field(
        tokenize=tokenize_en,
        pad_token=pad_word,
        lower=True
    )
    tgt_field = Field(
        tokenize=tokenize_en,
        init_token=bos_word, eos_token=eos_word, pad_token=pad_word,
        lower=True
    )

    # Create list of Examples from the list of sentences
    examples = []
    for sentence in sentences:
        sentence_split = sentence.split(' ')
        sentence_length = len(sentence_split)

        if sentence_length < min_length:
            continue

        src_length = min_length - 2
        for i in range(0, sentence_length - src_length, src_length):
            src = ' '.join(sentence_split[i:i + src_length])
            tgt = ' '.join(sentence_split[i + src_length:])

            example = Example.fromlist(
                data=[src, tgt],
                fields=[('src', src_field), ('tgt', tgt_field)]
            )
            examples.append(example)

    train_dataset, valid_dataset = Dataset(
        examples=examples,
        fields=[('src', src_field), ('tgt', tgt_field)]
    ).split(split_ratio=[0.9, 0.1])

    # Set the minimum frequency needed to include a token in the vocabulary
    min_freq = 2
    src_field.build_vocab(train_dataset, min_freq=min_freq)
    tgt_field.build_vocab(train_dataset, min_freq=min_freq)

    return train_dataset, valid_dataset, src_field, tgt_field


class SimpleIterator(Iterator):
    """
    A simple iterator that loads batches of data from a Dataset

    Arguments:
        dataset: the Dataset object to load Examples from
        batch_size: batch size
        sort_key: key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding
        device: 'cpu' or 'cuda' where variables are going to be created on
        train: whether the iterator is for a train set (influences shuffling
            and sorting)
    """
    def __init__(self, dataset, batch_size, sort_key, device, train):
        super().__init__(
            dataset=dataset, batch_size=batch_size, sort_key=sort_key,
            device=device, train=train
        )
        self.batches = []

    def create_batches(self):
        for b in batch(self.data(), self.batch_size, self.batch_size_fn):
            self.batches.append(sorted(b, key=self.sort_key))
