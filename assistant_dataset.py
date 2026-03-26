# dialog_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import json
from transformers import GPT2TokenizerFast, AutoModelForCausalLM



@dataclass
class AssistantConfig:
    max_length: int = 1024
    add_eos: bool = True          # add eos after every Assistant-answer

    token_user: str = "<|user|>"
    token_assistant: str = "<|assistant|>"
    token_knowledge: str = "<|knowledge|>"
    token_turn: str = "<|turn|>"


def parse_line(line: str) -> Tuple[int, str, str]:
    """
    "1 User: Hello"
    "2 Assistant: Hello!"
    """
    s = line.strip()
    sp = s.split(" ", 1)
    if len(sp) != 2 or not sp[0].isdigit():
        raise ValueError(f"Bad line (no leading number): {line!r}")
    n = int(sp[0])
    rest = sp[1]

    if rest.startswith("User:"):
        payload = rest[len("User:"):].lstrip()
        return n, "User", payload

    if rest.startswith("Assistant:"):
        payload = rest[len("Assistant:"):].lstrip()
        return n, "Assistant", payload

    raise ValueError(f"Bad line (expected 'User:' or 'Assistant:'): {line!r}")


# ------------------------- parsing -------------------------

def load_dialogs(files: list[str]) -> List[List[Tuple[str, str]]]:

    dialogs: List[List[Tuple[str, str]]] = []
    cur_pairs: List[Tuple[str, str]] = []

    last_n: Optional[int] = None
    pending_user: Optional[str] = None

    def flush_dialog():
        nonlocal cur_pairs, pending_user
        pending_user = None
        if cur_pairs:
            dialogs.append(cur_pairs)
        cur_pairs = []

    for file_name in files:
        with Path(file_name).open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line.strip():
                    # empty line = border of dialog
                    flush_dialog()
                    last_n = None
                    continue

                n, kind, payload = parse_line(line)
                # small filtering
                payload = payload.replace("’", "'")

                # bound of dialog by reset/decrement of index
                if last_n is not None and (n == 1 or n < last_n):
                    flush_dialog()

                last_n = n

                if kind == "User":
                    pending_user = payload
                else:
                    # assistant
                    if pending_user is None:
                        # if assistant without user - skip
                        continue
                    cur_pairs.append((pending_user, payload))
                    pending_user = None

        # file ended - close dialog
        flush_dialog()
        last_n = None

    return dialogs


class AssistantDataset(Dataset):
    """
    Обычные диалоги без табуляций:
      "<n> User: <text>"
      "<n> Assistant: <text>"

    Граница диалога:
      - индекс n сбрасывается на 1, или
      - пустая строка (если есть), или
      - индекс уменьшается

    1 диалог = 1 sample (как ChatGPT SFT по сути)
    Loss: mask all assistant => учим только контент ответов Assistant (включая eos, если add_eos=True).
    """

    def __init__(
        self,
        files: Union[str, List[str]],
        tokenizer,
        cfg: AssistantConfig = AssistantConfig(),
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg

        if isinstance(files, str):
            files = [files]
        self.files = [x for x in files]

        if self.cfg.add_eos and getattr(self.tokenizer, "eos_token", None) is None:
            raise ValueError("Tokenizer has no eos_token. Set add_eos=False or use tokenizer with eos_token.")

        self.dialogs = load_dialogs(self.files)
        self.samples = [self._make_text_and_parts(pairs) for pairs in self.dialogs]


    def save_to_jsonl(self, file_name: str):
        with open(file_name, "w", encoding="utf-8") as f:
            for dialog in self.dialogs:
                items = []
                for pair in dialog:
                    items.extend(pair)
                f.write(json.dumps(items, ensure_ascii=False) + "\n")


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        full_text, parts = self.samples[idx]

        enc_full = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.cfg.max_length,
            padding=False,
            return_attention_mask=True,
        )
        input_ids = torch.tensor(enc_full["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc_full["attention_mask"], dtype=torch.long)

        # Строим loss_mask через parts, без offset_mapping (работает и с slow токенизаторами)
        running_text = ""
        running_ids: List[int] = []
        loss_mask: List[int] = []

        for piece_text, is_answer in parts:
            running_text += piece_text

            #print(f"Tokens: {self.tokenizer.tokenize(running_text)}")

            enc = self.tokenizer(
                running_text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            new_ids = enc["input_ids"]
            delta = new_ids[len(running_ids):]
            running_ids = new_ids
            loss_mask.extend([1 if is_answer else 0] * len(delta))


        L = len(enc_full["input_ids"])
        lm = loss_mask[:L]

        labels = torch.full((L,), -100, dtype=torch.long)
        for i in range(L):
            if lm[i] == 1:
                labels[i] = input_ids[i]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": full_text
            }


    # ------------------------- building -------------------------

    def _make_text_and_parts(self, pairs: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, bool]]]:
        """
        parts = [(text_piece, is_answer)]
          - is_answer=True only for content of Assistant (plus eos, if included)
          - "User:" и "Assistant:" префиксы - is_answer=False (do not lern them)
        """
        eos = self.tokenizer.eos_token if (self.cfg.add_eos and self.tokenizer.eos_token) else ""


        parts: List[Tuple[str, bool]] = []
        full_chunks: List[str] = []

        for u, a in pairs:

            t_user = f"{self.cfg.token_user} {u}{self.cfg.token_assistant}"
            parts.append((t_user, False))
            full_chunks.append(t_user)

            # учим только контент ответа (+ eos), потом перевод строки
            t_ans = f"{a}{eos}"
            parts.append((t_ans, True))
            full_chunks.append(t_ans)

        return "".join(full_chunks), parts


def collate_lm_batch(
    batch: List[Dict[str, torch.Tensor]],
    padding_id: int,
    label_padding_id: int = -100
) -> Dict[str, torch.Tensor]:
    """
    batch: items list from DialogDataset, each item is:
      {"input_ids": 1D LongTensor, "attention_mask": 1D LongTensor, "labels": 1D LongTensor}

    padding_value: padding value for input_ids
    label_padding_value: should be -100 for causal LM
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")

    max_len = max(x["input_ids"].numel() for x in batch)

    bsize = len(batch)
    input_ids = torch.full((bsize, max_len), padding_id, dtype=torch.long)
    attention_mask = torch.zeros((bsize, max_len), dtype=torch.long)
    labels = torch.full((bsize, max_len), label_padding_id, dtype=torch.long)

    for i, x in enumerate(batch):
        ids = x["input_ids"]
        am = x["attention_mask"]
        lab = x["labels"]

        L = ids.numel()
        input_ids[i, :L] = ids
        attention_mask[i, :L] = am
        labels[i, :L] = lab

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


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

    dataset = AssistantDataset(["data/dialogues_clarification_64.txt"], tokenizer=tokenizer)
    #print(dataset.dialogs[0])
    full_text, parts = dataset.samples[0]

    print(parts)
