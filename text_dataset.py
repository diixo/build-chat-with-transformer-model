

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


def format_example(ex: dict, eos_token: str = "") -> str:
    parts = []

    if ex["knowledge"]:
        parts.append(f"<|knowledge|>\n{ex['knowledge']}")
    if ex["user"]:
        parts.append(f"<|user|>\n{ex['user']}")
    if ex["assistant"]:
        parts.append(f"<|assistant|>\n{ex['assistant']}{eos_token}")

    return "\n".join(parts)


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
