import argparse

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
        return tokenizer(example['lines'], truncation=True)

    def make_blocks(examples):
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        return result

    return dataset.map(process_example, remove_columns=["speaker", "lines"])\
        .map(make_blocks, batched=True, num_proc=4)


def train(wandb: bool = False, push_to_hub: bool = False):
    report_to = "wandb" if wandb else "none"

    data = script_dataset("input.txt")

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tokenizer = tokenizer.train_new_from_iterator(
        data["train"]["lines"], vocab_size=1000)

    data = preprocess(data, tokenizer)

    collate = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    model_config = BigramLanguageModelConfig(vocab_size=tokenizer.vocab_size)
    model = BigramLanguageModel(model_config)

    training_args = TrainingArguments(
        output_dir="bigram-model",
        report_to=report_to,
        learning_rate=2e-3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
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
    if push_to_hub:
        trainer.push_to_hub()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--push-to-hub",
                        action="store_true",
                        help="upload the trained model to Hugging Face")
    parser.add_argument("-r", "--report-to-wandb",
                        action="store_true",
                        help="Upload report to Weights and Biases")
    args = parser.parse_args()

    train(args.report_to_wandb, args.push_to_hub)


if __name__ == "__main__":
    train()
