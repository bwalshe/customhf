from itertools import takewhile, dropwhile
from pathlib import Path
import random
from typing import Optional

from datasets import Dataset
import numpy as np
from transformers import PreTrainedTokenizer, AutoTokenizer


class SlidingWindowDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, train: np.array, test: np.array):
        self._tokenizer = tokenizer
        self._train = train
        self._test = test

    def save(self, path):
        self._tokenizer.save_pretrained(path / "tokenizer")
        self._train.tofile(path / "train.bin")
        self._test.tofile(path / "test.bin")

    def train(self, block_size=16) -> Dataset:
        return sliding_window(self._train, block_size)

    def test(self, block_size=16) -> Dataset:
        return sliding_window(self._test, block_size)

    @property
    def tokenizer(self):
        return self._tokenizer

    @classmethod
    def load(cls, path: Path) -> "RandomSampleDataset":
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        train = np.memmap(path / "train.bin", dtype=np.uint64, mode="r")
        test = np.memmap(path / "test.bin", dtype=np.uint64, mode="r")
        return cls(tokenizer, train, test)


def make_dataset(
    tokenizer: PreTrainedTokenizer,
    text_file: Path,
    *,
    token_limit: Optional[int] = None,
    split_ratio: Optional[float] = 0.9,
):
    with text_file.open() as text_stream:
        text = text_stream.read()

    if token_limit is not None:
        tokenizer = tokenizer.train_new_from_iterator(
            [text], vocab_size=token_limit)
    tokens = tokenizer.encode(text, return_tensors="np")[0]

    split_point = int(len(tokens) * split_ratio)
    train = tokens[:split_point]
    test = tokens[split_point:]
    return SlidingWindowDataset(tokenizer, train, test)


def script_dataset(path: str, block_size: int = 16) -> Dataset:
    def gen():
        with open(path) as infile:
            lines = (line.strip() for line in infile)
            while True:
                try:
                    block = takewhile(bool, lines)
                    yield {"speaker": next(block), "lines": "\n".join(block)}
                    dropwhile(lambda line: not line, lines)
                except StopIteration:
                    return

    return Dataset.from_generator(gen).shuffle().train_test_split()


def sliding_window(data: np.array, window_size=16) -> Dataset:
    def gen():
        for i in range(len(data) - window_size + 1):
            yield {"input_ids": data[i: i + window_size]}
    return Dataset.from_generator(gen)


def random_sample_dataset(data: np.array, block_size=16) -> Dataset:
    def gen():
        while True:
            i = random.randint(0, len(data) - block_size - 1)
            yield {"input_ids": data[i: i + block_size]}

    return Dataset.from_generator(gen).shuffle()
