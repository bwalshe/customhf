import pytest
from pathlib import Path

import numpy as np
import transformers


import customhf
from customhf.data import make_dataset, sliding_window, SlidingWindowDataset


def test_sliding_window():
    raw_data = np.array(range(10))
    dataset = sliding_window(raw_data, 8)
    col_name = "input_ids"
    assert dataset.column_names == [col_name]
    assert len(dataset) == 3
    assert np.array_equal(dataset[0][col_name], raw_data[0:8])
    assert np.array_equal(dataset[1][col_name], raw_data[1:9])
    assert np.array_equal(dataset[2][col_name], raw_data[2:10])


def test_make_dataset(mocker, tmp_path):
    tokenizer = mocker.MagicMock(transformers.PreTrainedTokenizer)
    data = list(range(10))
    tokenizer.encode.return_value = [data]
    input_file = tmp_path / "input.txt"
    content = "content"
    input_file.write_text(content)
    mocker.patch("customhf.data.SlidingWindowDataset")
    make_dataset(tokenizer, input_file)
    tokenizer.encode.assert_called_with(content, return_tensors="np")
    customhf.data.SlidingWindowDataset.assert_called_with(
        tokenizer, data[:9], data[9:])

    # train = dataset.train()
    # assert train.column_names == ["text"]
