

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


def format_item(example: dict) -> str:
    knowledge="", user="", assistant=""

    if example["knowledge"]:
        knowledge = f"<|knowledge|>\n{example['knowledge']}"
    if example["user"]:
        user = f"<|user|>\n{example['user']}"
    if example["assistant"]:
        assistant = f"<|assistant|>\n{example['assistant']}"
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


    def __getitem__(self, index):

        raw_item = self.data[index]
        knowledge, user, assistant = format_item(raw_item)

        knowledge_ids = self.tokenizer(
            knowledge, truncation=True, add_special_tokens=False, max_length=(1024-1), return_tensors="pt")["input_ids"]
        

        if user != "" and assistant != "":

            qa_ids = self.tokenizer(
                user+assistant, truncation=True, add_special_tokens=False, max_length=(1024-1), return_tensors="pt")["input_ids"]
            
            # combine into one sequence, add eos_token_id at the end to prevent GPT2 from cutting the answer
            input_ids = torch.cat([
                knowledge_ids,                                                  # (1, N)
                qa_ids,                                                         # (1, M)
                torch.tensor([[self.tokenizer.eos_token_id]], dtype=torch.long) # (1, 1)
            ], dim=1)                                                           # (1, N+M+1)=shape([0],[1])
            # create new array
            labels = input_ids.clone()
        else:
            # use only knowledge
            input_ids = torch.cat([
                knowledge_ids,                                                  # (1, N)
                torch.tensor([[self.tokenizer.eos_token_id]], dtype=torch.long) # (1, 1)
            ], dim=1)                                                           # (1, N+M+1)=shape([0],[1])
            # create new array
            labels = input_ids.clone()


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
