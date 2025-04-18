import torch
from torch import nn, Tensor
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    GenerationMixin,
    AutoModelForCausalLM,
    AutoConfig)
from transformers.modeling_outputs import CausalLMOutput

from customhf import _logger


class BigramLanguageModelConfig(PretrainedConfig):
    model_type = "bigram-language"


class BigramLanguageModel(PreTrainedModel, GenerationMixin):
    config_class = BigramLanguageModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.pad_token_id = -100
        self.token_embedding_table = nn.Embedding(
            self.vocab_size, self.vocab_size)
        self.loss_type = "ForCausalLM"

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor = None,
                labels: Tensor = None,
                inputs_embeds=None,
                use_cache=False,
                return_dict=None
                ) -> CausalLMOutput | tuple[Tensor, Tensor]:
        """Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                ids of the input tokens
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
            inputs_embeds: needed for compatibilility wtih text generation pipeline, processing these inputs is not implemented
            use_cache: needed for compatiblity with text gneration pipeline - ignored.
            return_dict: (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        if inputs_embeds is not None:
            raise NotImplementedError(
                "BigramLanguageModel does not use input embeddings")

        if use_cache:
            _logger.info("BigramLanguageModel.forward() called with "
                         "use_cache=True, this will be ignored")

        if return_dict is None:
            return_dict = self.config.return_dict

        logits = self.token_embedding_table(input_ids)

        loss = None
        if labels is not None:
            labels = self._apply_mask(labels, attention_mask)
            loss = self.loss_function(
                logits, labels,
                vocab_size=self.vocab_size,
                ignore_index=self.pad_token_id)
        return self._format_results(loss, logits, return_dict)

    def _apply_mask(self, values: Tensor, mask: Tensor) -> Tensor:
        if mask is None or mask.all():
            return values
        mask = torch.ones_like(mask) - mask
        return values.masked_fill(mask.bool(), self.pad_token_id)

    @staticmethod
    def _format_results(loss, logits, return_dict):
        if return_dict is False:
            return logits, loss
        return CausalLMOutput(loss=loss, logits=logits)


def register_bigram_language_model():
    AutoConfig.register("bigram-language", BigramLanguageModelConfig)
    AutoModelForCausalLM.register(
        BigramLanguageModelConfig, BigramLanguageModel)
