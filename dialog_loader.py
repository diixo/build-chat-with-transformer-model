
from typing import List, Dict, Any, Optional, Tuple
import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass


@dataclass
class DialogConfig:

    max_len: int = 256

    token_pad: str = "<|pad|>"
    token_cls: str = "<|ctx|>"
    token_sep: str = "<|sep|>"


def read_jsonl_dataset(file_path: str):
    # read jsonl to list of tuples
    dialogs = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            dialog = json.loads(line)
            dialogs.append(tuple(dialog))
    return dialogs


# Speakers dialogues dataset loader
'''
all_turns_tokens = [CLS] + turn0 + [SEP] + turn1 + [SEP] + turn2 + [SEP] + ...
label_tokens = [PAD] + PAD(turn0) + turn1 + PAD(turn2) + turn3 + ...

CrossEntropyLoss(ignore_index=pad_token_id), instead of IGNORE_INDEX = -100:

* ignores CLS
* ignores user-turns
* ignores padding
* calculates loss only for assistant-turns
'''
class DialogLoader(Dataset):
    IGNORE_INDEX = -100

    def __init__(self, data: List[Tuple[str, ...]], tokenizer, config: DialogConfig):
        self.tokenizer = tokenizer
        self.max_len = config.max_len

        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id must not be None")
        if self.tokenizer.cls_token_id is None:
            raise ValueError("tokenizer.cls_token_id must not be None")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must not be None")

        self.turn_sep_id = tokenizer.sep_token_id
        self.ctx_token_id = tokenizer.cls_token_id
        self.pad_token_id = tokenizer.pad_token_id

        self.data = []
        for dialog in data:
            self.data.extend([dialog, dialog])

        self.length = len(self.data)


    def __len__(self):
        return self.length


    def make_data(self, multi_turn_sentences: Tuple[str, ...], predict_parity: int):
        #for i, sent in enumerate(multi_turn_sentences):
        #   print(f"Tokens[{i}]: {self.tokenizer.tokenize(sent)}")

        # ctx_token only once: at the very beginning of the whole dialogue
        input_ids = []
        labels = []

        for i, sentence in enumerate(multi_turn_sentences):
            sentence_ids = self.tokenizer(
                sentence,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"].squeeze(0)

            turn_ids = sentence_ids.tolist() + [self.turn_sep_id]

            if i % 2 == predict_parity:
                labels.extend(turn_ids)
            else:
                # masked input = <|user|> utterance <|assistant|>
                turn_ids = [self.ctx_token_id] + turn_ids
                labels.extend([self.IGNORE_INDEX] * len(turn_ids))

            # append first, then trim if needed
            input_ids.extend(turn_ids)

            if len(input_ids) > self.max_len:
                input_ids = input_ids[:self.max_len]
                labels = labels[:self.max_len]
                break

            if len(input_ids) == self.max_len:
                break

        if len(input_ids) < self.max_len:
            if input_ids[-1] == self.turn_sep_id:
                input_ids[-1] = self.tokenizer.eos_token_id
                labels[-1] = self.tokenizer.eos_token_id
            else:
                input_ids.append(self.tokenizer.eos_token_id)
                labels.append(self.tokenizer.eos_token_id)
        else:
            if input_ids[-1] == self.turn_sep_id:
                input_ids[-1] = self.tokenizer.eos_token_id
                labels[-1] = self.tokenizer.eos_token_id


        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


    def __getitem__(self, idx: int):

        multi_turn_sentences = self.data[idx]
        predict_parity = idx % 2

        input_ids, labels = self.make_data(
            multi_turn_sentences=multi_turn_sentences,
            predict_parity=predict_parity
        )
        return { "input_ids": input_ids, "labels": labels }



def collate_fn_batch(
    batch: List[Dict[str, torch.Tensor]],
    padding_id: int,
    label_padding_id: int = -100
) -> Dict[str, torch.Tensor]:

    max_len = max(item["input_ids"].size(0) for item in batch)

    batch_input_ids = []
    batch_labels = []

    for item in batch:
        input_ids = item["input_ids"]
        labels = item["labels"]

        pad_len = max_len - input_ids.size(0)

        if pad_len > 0:
            input_ids = torch.cat(
                [input_ids, torch.full((pad_len,), padding_id, dtype=torch.long)],
                dim=0
            )
            labels = torch.cat(
                [labels, torch.full((pad_len,), label_padding_id, dtype=torch.long)],
                dim=0
            )

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)

    return {
        "input_ids": torch.stack(batch_input_ids, dim=0),
        "labels": torch.stack(batch_labels, dim=0),
    }
