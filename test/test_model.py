import torch
from torch import nn

from customhf.model import BigramLanguageModel, BigramLanguageModelConfig


def test_loss_fn():
    vocab_size = 2
    batch_size = 5
    input_len = 10
    ignore_index = -100

    config = BigramLanguageModelConfig(vocab_size=vocab_size)
    model = BigramLanguageModel(config)

    input_ids = torch.randint(vocab_size, (batch_size, input_len))
    targets = input_ids.clone()

    result = model(input_ids=input_ids, attention_mask=None, labels=targets)
    loss = result["loss"]
    logits = result["logits"].float()

    B, T, C = logits.shape

    logits = logits.view(B * T, C)

    targets = nn.functional.pad(targets, (0, 1), value=ignore_index)
    targets = targets[..., 1:].contiguous()
    targets = targets.view(B * T)
    karpathy_loss = nn.functional.cross_entropy(
        logits, targets, ignore_index=ignore_index)

    assert loss == karpathy_loss
