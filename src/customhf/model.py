from torch import nn

from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.modeling_outputs import CausalLMOutput


class BigramLanguageModelConfig(PretrainedConfig):
    model_type = "bigram-language"


class BigramLanguageModel(PreTrainedModel, GenerationMixin):
    config_class = BigramLanguageModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.token_embedding_table = nn.Embedding(
            self.vocab_size, self.vocab_size)
        self.loss_type = "ForCausalLM"

    def forward(self, input_ids, labels=None, **kwargs):
        """Args:
            idx: (B,T) tensor of integers
            targets: (B,T) tensor of integers
        """
        logits = self.token_embedding_table(input_ids)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits, labels, vocab_size=self.vocab_size)
        return CausalLMOutput(loss=loss, logits=logits)


MODEL_FOR_CAUSAL_LM_MAPPING.register("bigram-language", BigramLanguageModel)
