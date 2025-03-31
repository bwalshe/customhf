from itertools import takewhile, dropwhile

from datasets import Dataset


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
