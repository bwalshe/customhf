import argparse

from customhf.model import BigramLanguageModel, BigramLanguageModelConfig
from transformers import pipeline, AutoTokenizer


def main(text: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    config = BigramLanguageModelConfig(vocab_size=tokenizer.vocab_size)
    model = BigramLanguageModel(config)
    generate = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = generate(text)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('text')
    args = parser.parse_args()
    main(args.text)
