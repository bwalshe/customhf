from torch import nn, Tensor

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    GenerationMixin,
    AutoModelForCausalLM,
    AutoConfig)
from transformers.modeling_outputs import CausalLMOutput


class BigramLanguageModelConfig(PretrainedConfig):
    model_type = "bigram-language"


class BigramLanguageModel(PreTrainedModel, GenerationMixin):
    config_class = BigramLanguageModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.pad_token_id = -100
        self.token_embedding_table = nn.Embedding(
            self.vocab_size, self.vocab_size, padding_idx=self.pad_token_id)
        self.loss_type = "ForCausalLM"

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor = None,
                labels: Tensor = None,
                **kwargs) -> CausalLMOutput:
        """Args:
            idx: (B,T) tensor of integers
            targets: (B,T) tensor of integers
        """
        input_ids = self._apply_mask(input_ids, attention_mask)
        logits = self.token_embedding_table(input_ids)

        loss = None
        if labels is not None:
            labels = self._apply_mask(labels, attention_mask)
            loss = self.loss_function(
                logits, labels, vocab_size=self.vocab_size, ignore_index=self.pad_token_id)
        return CausalLMOutput(loss=loss, logits=logits)

    def _apply_mask(self, values: Tensor, mask: Tensor) -> Tensor:
        if mask.all():
            return values
        return values.masked_fill(mask.bool(), self.pad_token_id)


AutoConfig.register("bigram-language", BigramLanguageModelConfig)
AutoModelForCausalLM.register(BigramLanguageModelConfig, BigramLanguageModel)
