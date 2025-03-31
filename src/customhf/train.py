from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from customhf.data import script_dataset
from customhf.model import BigramLanguageModel, BigramLanguageModelConfig


def preprocess(dataset: Dataset, tokenizer: PreTrainedTokenizer, block_size: int = 8) -> Dataset:

    def process_example(example: dict[str, str]):
        return tokenizer(example['lines'], truncation=True,
                         max_length=block_size+1)

    return dataset.map(process_example, remove_columns=["speaker", "lines"])


def train():
    data = script_dataset("input.txt")

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tokenizer = tokenizer.train_new_from_iterator(
        data["train"]["lines"], vocab_size=1000)

    data = preprocess(data, tokenizer)

    # tokenizer.pad_token = tokenizer.eos_token
    collate = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    model_config = BigramLanguageModelConfig(vocab_size=tokenizer.vocab_size)
    model = BigramLanguageModel(model_config)

    training_args = TrainingArguments(
        output_dir="bigram-model",
        report_to="none",
        learning_rate=2e-3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        processing_class=tokenizer,
        data_collator=collate
    )

    trainer.train()


if __name__ == "__main__":
    train()
