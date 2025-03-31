from torch import nn

from transformers import PretrainedConfig, PreTrainedModel


class BigramLanguageModelConfig(PretrainedConfig):
    model_type = "bigram_language"

    def __init__(self, vocab_size: int = 0, **kwargs):
        self.vocab_size = vocab_size
        super().__init__(**kwargs)


class BigramLanguageModel(PreTrainedModel):
    config_class = BigramLanguageModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.token_embedding_table = nn.Embedding(
            self.vocab_size, self.vocab_size)
        self.loss_type = "ForCausalLM"

    def forward(self, input_ids, attention_mask, labels=None):
        """Args:
            idx: (B,T) tensor of integers
            targets: (B,T) tensor of integers
        """
        logits = self.token_embedding_table(input_ids)

        if labels is None:
            return {"logits": logits}

        loss = self.loss_function(logits, labels, vocab_size=self.vocab_size)
        return {"logits": logits, "loss": loss}
