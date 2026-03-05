
from dialog_dataset import DialogDataset, DialogConfig

from transformers import GPT2TokenizerFast


if __name__ == "__main__":

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    dataset = DialogDataset("data/test.txt", tokenizer)

    print(len(dataset))
