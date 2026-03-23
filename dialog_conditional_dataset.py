
from typing import Dict, List, Any, Optional
import torch

import json
from torch.utils.data import Dataset
from dialog_dataset import DialogConfig
from transformers import GPT2TokenizerFast


class DialogConditionalDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: Optional[int] = None,
        add_eos: bool = False,
        cfg: DialogConfig = DialogConfig(),
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_eos = add_eos

        self.tok_knowledge = cfg.token_knowledge
        self.tok_user = cfg.token_user
        self.tok_assistant = cfg.token_assistant
        self.tok_turn = cfg.token_turn

        self.samples = self._load_file(file_path)


    def _load_file(self, file_path: str) -> List[Dict[str, Any]]:
        if file_path.endswith(".jsonl"):
            samples = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    samples.append(json.loads(line))
            return samples

        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                raise ValueError("JSON root must be list or dict")

        else:
            raise ValueError("Only .jsonl or .json are supported")


    def _knowledge_to_text(self, knowledge: Dict[str, Any]) -> str:
        if not knowledge:
            return ""

        lines = []
        for key, value in knowledge.items():
            if value is None:
                value = ""
            lines.append(f"{key}={value}")
        return "\n".join(lines)


    def _dialog_to_text(self, dialog: List[Dict[str, str]]) -> str:
        parts = []

        for msg in dialog:
            role = msg["role"].strip().lower()
            content = msg["content"]

            if role == "user":
                role_token = self.tok_user
            elif role == "assistant":
                role_token = self.tok_assistant
            else:
                raise ValueError(f"Unsupported role: {role}")

            parts.append(role_token)
            parts.append("\n")
            parts.append(content)
            parts.append("\n")
            parts.append(self.tok_turn)
            parts.append("\n")

        return "".join(parts)


    def _build_texts(self, sample: Dict[str, Any]):
        knowledge_in = self._knowledge_to_text(sample.get("knowledge_in", {}))
        dialog_text = self._dialog_to_text(sample["dialog"])
        knowledge_out = self._knowledge_to_text(sample.get("knowledge_out", {}))
        assistant_text = sample["assistant"]

        prefix_text = (
            f"{self.tok_knowledge}\n"
            f"{knowledge_in}\n"
            f"{self.tok_turn}\n"
            f"{dialog_text}"
        )

        target_text = (
            f"{self.tok_knowledge}\n"
            f"{knowledge_out}\n"
            f"{self.tok_turn}\n"
            f"{self.tok_assistant}\n"
            f"{assistant_text}\n"
            f"{self.tok_turn}"
        )

        return prefix_text, target_text


    def _encode_sample(self, sample: Dict[str, Any]) -> Dict[str, List[int]]:
        prefix_text, target_text = self._build_texts(sample)

        prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]

        input_ids = prefix_ids + target_ids
        labels = [-100] * len(prefix_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        if self.add_eos and self.tokenizer.eos_token_id is not None:
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)
            attention_mask.append(1)

        if self.max_length is not None:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self._encode_sample(sample)


def collate_lm_batch(
    batch: List[Dict[str, List[int]]],
    padding_value: int,
    label_padding_value: int = -100,
):
    """
    Паддинг батча для causal LM.
    batch: список dict с ключами input_ids, attention_mask, labels
    """
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        input_ids.append(item["input_ids"] + [padding_value] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [label_padding_value] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


if __name__ == "__main__":

    tokenizer = GPT2TokenizerFast.from_pretrained(
        "gpt2",
        local_files_only=False,
        padding_side="right",
        model_max_length=1024
        )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id to use as pad.")
        tokenizer.pad_token = tokenizer.eos_token


    dataset = DialogConditionalDataset(
        file_path="test-dialog.json",
        tokenizer=tokenizer,
        max_length=256,
        add_eos=False,
    )

    item = dataset[0]
    print(item.keys())
    print(len(item["input_ids"]))
    print(len(item["labels"]))
