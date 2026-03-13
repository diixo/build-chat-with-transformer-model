

from typing import List

import torch
from torch.utils.data import Dataset

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from transformers import GPT2TokenizerFast


@dataclass
class DialogConfig:
    max_length: int = 1024
    add_eos: bool = True          # добавлять eos после каждого Assistant-ответа
    line_sep: str = "\n"          # разделитель строк в склеенном тексте


from typing import List, Dict, Optional


from typing import List, Dict, Optional


def load_text(path: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    current_user: Optional[str] = None
    pending_knowledge: Optional[str] = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("User:"):
                current_user = line[len("User:"):].strip()

            elif line.startswith("Assistant:"):
                assistant_text = line[len("Assistant:"):].strip()

                if current_user is not None:
                    items.append({
                        "knowledge": pending_knowledge if pending_knowledge is not None else assistant_text,
                        "user": current_user,
                        "assistant": assistant_text,
                    })

                current_user = None
                pending_knowledge = None

            else:
                # любая обычная строка = отдельная knowledge-запись
                items.append({
                    "knowledge": line,
                })

                # и она же становится knowledge для следующей пары User/Assistant
                pending_knowledge = line

    return items


def format_item(example: dict):
    knowledge = ""
    user = ""
    assistant = ""

    if example.get("knowledge"):
        knowledge = f"<|knowledge|>\n{example['knowledge']}\n"
    if example.get("user"):
        user = f"<|user|>\n{example['user']}\n"
    if example.get("assistant"):
        assistant = f"<|assistant|>\n{example['assistant']}\n"
    return knowledge, user, assistant


class TextDataset(Dataset):

    def __init__(
        self,
        files: Union[str, Path, List[Union[str, Path]]],
        tokenizer,
        cfg: DialogConfig = DialogConfig(),
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg

        if isinstance(files, (str, Path)):
            files = [files]
        self.files = [Path(x) for x in files]

        if self.cfg.add_eos and getattr(self.tokenizer, "eos_token", None) is None:
            raise ValueError("Tokenizer has no eos_token. Set add_eos=False or use tokenizer with eos_token.")

        self.items: List[str] = []
        for fp in self.files:
            self.items.extend(load_text(str(fp)))


    def __len__(self):
        return len(self.data)


    def get_item_str(self, index) -> str:
        knowledge, user, assistant = format_item(self.items[index])
        return knowledge + user + assistant


    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        raw_item = self.items[index]

        knowledge, user, assistant = format_item(raw_item)

        parts_input_ids = []
        parts_labels = []

        # 1. knowledge
        if knowledge:
            knowledge_ids = self.tokenizer(
                knowledge,
                truncation=True,
                add_special_tokens=False,
                max_length=self.cfg.max_length,
                return_tensors="pt"
            )["input_ids"].squeeze(0)   # (seq,)
            parts_input_ids.append(knowledge_ids)
            parts_labels.append(torch.full_like(knowledge_ids, -100))

        # 2. user
        if user:
            user_ids = self.tokenizer(
                user,
                truncation=True,
                add_special_tokens=False,
                max_length=self.cfg.max_length,
                return_tensors="pt"
            )["input_ids"].squeeze(0)
            parts_input_ids.append(user_ids)
            parts_labels.append(torch.full_like(user_ids, -100))

        # 3. assistant
        if assistant:
            assistant_ids = self.tokenizer(
                assistant,
                truncation=True,
                add_special_tokens=False,
                max_length=self.cfg.max_length,
                return_tensors="pt"
            )["input_ids"].squeeze(0)
            parts_input_ids.append(assistant_ids)
            parts_labels.append(assistant_ids.clone())

        # 4. eos
        if self.cfg.add_eos:
            eos_id = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            parts_input_ids.append(eos_id)

            if assistant:
                parts_labels.append(eos_id.clone())
            else:
                parts_labels.append(torch.tensor([-100], dtype=torch.long))

        # fallback: если вдруг всё пустое
        if not parts_input_ids:
            eos_id = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            input_ids = eos_id
            labels = torch.tensor([-100], dtype=torch.long)
        else:
            input_ids = torch.cat(parts_input_ids, dim=0)
            labels = torch.cat(parts_labels, dim=0)

        # обрезка до max_length
        max_len = self.cfg.max_length
        if input_ids.size(0) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_batch(batch, padding_value, label_padding_value=-100):
    from torch.nn.utils.rnn import pad_sequence
    from collections import defaultdict

    new_batch = defaultdict(lambda:[])
    for x in batch:
        for x_key in x.keys():
            new_batch[x_key].append(x[x_key][0])

    new_batch = dict(new_batch)
    for batch_key in new_batch.keys():
        if batch_key == "labels":
            new_batch[batch_key] = pad_sequence(new_batch[batch_key], batch_first=True, padding_value=label_padding_value)
        else:
            new_batch[batch_key] = pad_sequence(new_batch[batch_key], batch_first=True, padding_value=padding_value)

    if "input_ids" in new_batch:
        new_batch["attention_mask"] = (new_batch["input_ids"] != padding_value).long()

    return new_batch


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

    dataset = TextDataset(["test.txt"], tokenizer=tokenizer)
    print(dataset.items)

    print(dataset[0])

    print(f"Tokens: {tokenizer.tokenize(dataset.get_item_str(0))}")
