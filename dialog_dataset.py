
# babi_dialog_dataset_v2.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


@dataclass
class DialogConfig:
    max_length: int = 1024
    add_eos: bool = True                  # добавить eos после целевого ответа (последнего)
    label_last_only: bool = True          # True: лосс только на последнем Assistant в sample
    keep_original_tabs: bool = True       # сохранять \t как в исходнике (обычно True)
    line_sep: str = "\n"                  # разделитель строк в склеенном тексте


class DialogDataset(Dataset):
    """
    Читает текстовые файлы bAbI-стиля, где строки:
      "<n> User: <text>"
      "<n> Assistant: <text>"

    Граница диалога:
      - индекс n сбрасывается на 1, или
      - пустая строка (если есть), или
      - индекс уменьшается

    Генерация сэмплов:
      Для диалога из K пар (User+Assistant) создаёт K сэмплов:
        (1 пара), (1..2 пары), ..., (1..K пар)
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

        if getattr(self.tokenizer, "eos_token_id", None) is None and self.cfg.add_eos:
            raise ValueError("Tokenizer has no eos_token_id/eos_token. Set add_eos=False or provide tokenizer with eos.")

        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        dialogs = self._load_dialogs()
        self.samples = self._build_samples(dialogs)


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        # Токенизируем полный текст (контекст + target)
        enc_full = self.tokenizer(
            s["full_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.cfg.max_length,
            padding=False,
            return_attention_mask=True,
        )
        input_ids = torch.tensor(enc_full["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc_full["attention_mask"], dtype=torch.long)

        labels = torch.full_like(input_ids, -100)

        if s["mode"] == "last_only":
            # Маска: -100 на префиксе, target токены = input_ids
            enc_prefix = self.tokenizer(
                s["prefix_text"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.cfg.max_length,
                padding=False,
                return_attention_mask=False,
            )
            prefix_len = len(enc_prefix["input_ids"])

            if prefix_len < len(input_ids):
                labels[prefix_len:] = input_ids[prefix_len:]

        else:
            # "all_assistant": лосс на всех Assistant-ответах в этом sample.
            # Делается как сумма сегментов: каждый сегмент = (prefix_up_to_answer, answer)
            # Тут проще собрать "loss mask" через отдельные границы в токенах.
            # Мы храним границы ответов в символах не будем; сделаем через посимвольный сбор:
            # последовательно токенизируем нарастающий текст и отмечаем ranges.
            # (работает с обычными fast/slow токенизаторами)
            full_ids = enc_full["input_ids"]
            # пересоберём токены по сегментам без truncation, потом обрежем как full_ids
            running_text = ""
            running_ids: List[int] = []
            loss_mask: List[int] = []

            for part_text, is_answer in s["parts"]:
                running_text += part_text
                enc = self.tokenizer(
                    running_text,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_attention_mask=False,
                )
                new_ids = enc["input_ids"]
                # новые токены = хвост после уже добавленных
                delta = new_ids[len(running_ids):]
                running_ids = new_ids

                loss_mask.extend([1 if is_answer else 0] * len(delta))

            # теперь применим truncation как в enc_full:
            # full_ids == running_ids[:len(full_ids)] (при условии одинаковой токенизации)
            L = len(full_ids)
            lm = loss_mask[:L]
            for i in range(L):
                if lm[i] == 1:
                    labels[i] = input_ids[i]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


    # ------------------------- parsing -------------------------

    def _load_dialogs(self) -> List[List[Tuple[str, str]]]:

        dialogs: List[List[Tuple[str, str]]] = []
        cur_pairs: List[Tuple[str, str]] = []

        last_n: Optional[int] = None
        pending_user: Optional[str] = None

        def flush_dialog():
            nonlocal cur_pairs, pending_user
            if pending_user is not None:
                # незакрытый user без assistant — можно либо дропнуть, либо сохранить пустым assistant
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
                        pending_user = None

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


    def _parse_line(self, line: str) -> Optional[Tuple[int, str, str]]:
        line = line.strip()
        if not line:
            return None

        # ожидаем: "<num> <Role>: <text>"
        # 1) отделяем номер
        parts = line.split(" ", 1)
        if len(parts) != 2:
            return None  # или raise ValueError(...)
        num_str, rest = parts[0], parts[1].strip()

        if not num_str.isdigit():
            return None
        turn_id = int(num_str)

        # 2) отделяем роль и текст по первому двоеточию
        if ":" not in rest:
            return None
        role, text = rest.split(":", 1)
        role = role.strip()
        text = text.strip()

        if role not in ("User", "Assistant"):
            return None

        return turn_id, role, text


    # ------------------------- sample building -------------------------

    def _build_samples(self, dialogs: List[List[Tuple[str, str]]]) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        for pairs in dialogs:
            # наращиваем контекст: 1 пара, 2 пары, ..., K пар
            history_pairs: List[Tuple[str, str]] = []
            for (u, a) in pairs:
                history_pairs.append((u, a))

                if self.cfg.label_last_only:
                    prefix_text, target_text, full_text = self._make_last_only_text(history_pairs)
                    samples.append(
                        {
                            "mode": "last_only",
                            "prefix_text": prefix_text,
                            "full_text": full_text,
                        }
                    )
                else:
                    # в parts: (text_piece, is_answer_piece)
                    full_text, parts = self._make_all_assistant_text(history_pairs)
                    samples.append(
                        {
                            "mode": "all_assistant",
                            "full_text": full_text,
                            "parts": parts,
                        }
                    )

        return samples

    def _make_last_only_text(self, pairs: List[Tuple[str, str]]) -> Tuple[str, str, str]:
        """
        Делает sample, где учим ТОЛЬКО последний ответ ассистента.
        prefix_text = весь диалог до "Assistant:\t" последней реплики (включая этот маркер)
        target_text = "<answer>\t0" (+ eos если включено)
        full_text = prefix + target
        """
        sep = self.cfg.line_sep

        # всё до последней пары полностью
        prefix_lines: List[str] = []
        for (u, a) in pairs[:-1]:
            prefix_lines.append(f"User: {u}")
            prefix_lines.append(f"Assistant:\t{a}\t0")

        # последняя пара: добавим User строку и маркер Assistant:\t в prefix
        last_u, last_a = pairs[-1]
        prefix_lines.append(f"User: {last_u}")
        prefix_lines.append("Assistant:\t")  # важно: таб сразу после двоеточия

        prefix_text = sep.join(prefix_lines)
        if not prefix_text.endswith("\t"):
            prefix_text += "\t"  # на всякий

        target_text = f"{last_a}\t0"
        if self.cfg.add_eos and self.eos_token_id is not None:
            # добавляем eos как токен, но строкой не всегда можно.
            # Самый надёжный способ: добавить eos на уровне input_ids в collate — но мы делаем проще:
            # если tokenizer.eos_token существует строкой, добавим его.
            eos_tok = getattr(self.tokenizer, "eos_token", None)
            if eos_tok:
                target_text += eos_tok

        full_text = prefix_text + target_text
        return prefix_text, target_text, full_text

    def _make_all_assistant_text(self, pairs: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, bool]]]:
        """
        Делает sample, где лосс на всех assistant-ответах (если тебе так надо).
        Возвращает:
          full_text
          parts: список (кусок_текста, это_ответ_ассистента?)
        """
        sep = self.cfg.line_sep

        parts: List[Tuple[str, bool]] = []
        lines: List[str] = []

        for (u, a) in pairs:
            lines.append(f"User: {u}{sep}")
            parts.append((f"User: {u}{sep}", False))

            # Assistant маркер (не учим)
            lines.append("Assistant:\t")
            parts.append(("Assistant:\t", False))

            # Answer + \t0 (+ eos) (учим)
            ans = f"{a}\t0"
            eos_tok = getattr(self.tokenizer, "eos_token", None)
            if self.cfg.add_eos and eos_tok:
                ans += eos_tok
            ans += sep
            lines.append(ans)
            parts.append((ans, True))

        full_text = "".join(lines)
        return full_text, parts


def make_lm_collate_fn(tokenizer, pad_to_multiple_of: Optional[int] = None):
    """
    Паддинг для dict{'input_ids','attention_mask','labels'}.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # часто для GPT pad делают = eos
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id to use as pad.")
        pad_id = tokenizer.eos_token_id

    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(x["input_ids"].numel() for x in batch)
        if pad_to_multiple_of:
            r = max_len % pad_to_multiple_of
            if r != 0:
                max_len += (pad_to_multiple_of - r)

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, x in enumerate(batch):
            L = x["input_ids"].numel()
            input_ids[i, :L] = x["input_ids"]
            attention_mask[i, :L] = x["attention_mask"]
            labels[i, :L] = x["labels"]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate
