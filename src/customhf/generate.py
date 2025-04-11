import argparse
from pathlib import Path

from customhf.model import register_bigram_language_model
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


def generate(text: str, model: Path) -> None:
    register_bigram_language_model()
    tokenizer = AutoTokenizer.from_pretrained(model, sep_token=None)
    model = AutoModelForCausalLM.from_pretrained(model)
    generate = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = generate(text, do_sample=True)
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=Path, required=True,
                        help="Path of a saved Hugging Face causal LM model")
    parser.add_argument('text', help="The inital text")
    args = parser.parse_args()
    generate(args.text, args.model)
