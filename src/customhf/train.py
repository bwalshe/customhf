import argparse

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from customhf.data import script_dataset
from customhf.bigram_model import BigramLanguageModel, BigramLanguageModelConfig
from customhf.gpt2_model import GPT, GPTConfig
from customhf import _try_setup_logging


def preprocess(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, block_size: int = 8
) -> Dataset:
    def process_example(example: dict[str, str]):
        return tokenizer(example["lines"], truncation=True)

    def make_blocks(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        return result

    return dataset.map(process_example, remove_columns=["speaker", "lines"]).map(
        make_blocks, batched=True, num_proc=4
    )


def make_model(name: str, tokenizer) -> PreTrainedModel:
    if name == "bigram-language":
        model_config = BigramLanguageModelConfig(vocab_size=tokenizer.vocab_size)
        return BigramLanguageModel(model_config)
    if name == "nano-gpt":
        model_config = GPTConfig(bias=False)
        return GPT(model_config)

    raise ValueError(f"Unknown model name: {name}")


def train(
    model_name: str,
    output_dir: str = "model-output",
    num_epochs: int = 10,
    wandb: bool = False,
    push_to_hub: bool = False,
):
    report_to = "wandb" if wandb else "none"

    data = script_dataset("input.txt")

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tokenizer = tokenizer.train_new_from_iterator(
        data["train"]["lines"], vocab_size=1000
    )

    data = preprocess(data, tokenizer)

    collate = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model = make_model(model_name, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to=report_to,
        learning_rate=2e-3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        push_to_hub=False,
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        processing_class=tokenizer,
        data_collator=collate,
    )

    trainer.train()
    if push_to_hub:
        trainer.push_to_hub()


def main():
    _try_setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name",
        choices=["bigram-language", "nano-gpt"],
        default="bigram-language",
        help="The name of the model to use",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="model_output",
        help="Location to store model checkpoints.",
    )
    parser.add_argument(
        "-n", "--num-epochs", default=10, help="Maximum number of epochs to run."
    )
    parser.add_argument(
        "-p",
        "--push-to-hub",
        action="store_true",
        help="Upload the trained model to Hugging Face.",
    )
    parser.add_argument(
        "-r",
        "--report-to-wandb",
        action="store_true",
        help="Upload report to Weights and Biases.",
    )
    args = parser.parse_args()

    train(
        args.model_name,
        args.output_dir,
        args.num_epochs,
        args.report_to_wandb,
        args.push_to_hub,
    )


if __name__ == "__main__":
    train()
