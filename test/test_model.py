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

    result = model(input_ids=input_ids, labels=targets)
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


def test_attention_mask():
    vocab_size = 2
    token_len = 10
    config = BigramLanguageModelConfig(vocab_size=vocab_size)
    model_short = BigramLanguageModel(config)
    model_long = BigramLanguageModel(config)

    with torch.no_grad():
        model_short.token_embedding_table.weight = torch.nn.Parameter(
            torch.eye(vocab_size))
        model_long.token_embedding_table.weight = torch.nn.Parameter(
            torch.eye(vocab_size))

    input_ids = torch.ones(1, token_len, dtype=torch.long)
    input_ids_short = input_ids[:, :token_len//2].clone()
    labels_short = input_ids_short.clone()

    labels_long = torch.cat((input_ids_short, 1-input_ids_short), 1)
    mask = labels_long.clone()

    loss_short = model_short(input_ids=input_ids_short,
                             labels=labels_short).loss
    loss_long = model_long(input_ids=input_ids,
                           attention_mask=mask, labels=labels_long).loss

    loss_short.backward()
    loss_long.backward()

    grad_short = model_short.token_embedding_table.weight.grad
    grad_long = model_long.token_embedding_table.weight.grad
    assert grad_long is not None
    assert torch.all(torch.eq(grad_short, grad_long))
