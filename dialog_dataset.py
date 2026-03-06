# dialog_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


@dataclass
class DialogConfig:
    max_length: int = 1024
    add_eos: bool = True          # добавлять eos после каждого Assistant-ответа
    line_sep: str = "\n"          # разделитель строк в склеенном тексте


class DialogDataset(Dataset):
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

        dialogs = self._load_dialogs()
        self.samples = [self._make_text_and_parts(pairs) for pairs in dialogs]


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

    # ------------------------- parsing -------------------------

    def _load_dialogs(self) -> List[List[Tuple[str, str]]]:
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

        for fp in self.files:
            with fp.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.rstrip("\n")
                    if not line.strip():
                        # пустая строка = граница диалога
                        flush_dialog()
                        last_n = None
                        continue

                    n, kind, payload = self._parse_line(line)

                    # граница диалога по сбросу/уменьшению индекса
                    if last_n is not None and (n == 1 or n < last_n):
                        flush_dialog()

                    last_n = n

                    if kind == "User":
                        pending_user = payload
                    else:
                        # assistant
                        if pending_user is None:
                            # если ассистент без юзера — пропустим (или можно считать user пустым)
                            continue
                        cur_pairs.append((pending_user, payload))
                        pending_user = None

            # файл закончился — закроем диалог
            flush_dialog()
            last_n = None

        return dialogs


    def _parse_line(self, line: str) -> Tuple[int, str, str]:
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


    # ------------------------- building -------------------------

    def _make_text_and_parts(self, pairs: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, bool]]]:
        """
        parts = [(text_piece, is_answer)]
          - is_answer=True only for content of Assistant (plus eos, if included)
          - "User:" и "Assistant:" префиксы - is_answer=False (do not lern them)
        """
        sep = self.cfg.line_sep
        eos = self.tokenizer.eos_token if (self.cfg.add_eos and self.tokenizer.eos_token) else ""

        parts: List[Tuple[str, bool]] = []
        full_chunks: List[str] = []

        for u, a in pairs:
            t_user = f"User: {u}{sep}"
            parts.append((t_user, False))
            full_chunks.append(t_user)

            t_aprefix = "Assistant: "
            parts.append((t_aprefix, False))
            full_chunks.append(t_aprefix)

            # учим только контент ответа (+ eos), потом перевод строки
            t_ans = f"{a}{eos}{sep}"
            parts.append((t_ans, True))
            full_chunks.append(t_ans)

        return "".join(full_chunks), parts


def collate_lm_batch(
    batch: List[Dict[str, torch.Tensor]],
    padding_value: int,
    label_padding_value: int = -100
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
    input_ids = torch.full((bsize, max_len), padding_value, dtype=torch.long)
    attention_mask = torch.zeros((bsize, max_len), dtype=torch.long)
    labels = torch.full((bsize, max_len), label_padding_value, dtype=torch.long)

    for i, x in enumerate(batch):
        ids = x["input_ids"]
        am = x["attention_mask"]
        lab = x["labels"]

        L = ids.numel()
        input_ids[i, :L] = ids
        attention_mask[i, :L] = am
        labels[i, :L] = lab

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
