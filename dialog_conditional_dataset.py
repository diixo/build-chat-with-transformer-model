
from typing import Dict, List, Any, Optional
import torch


SPECIAL_TOKENS = {
    "knowledge": "<knowledge>",
    "user": "<user>",
    "assistant": "<assistant>",
    "turn": "<turn>",
}


def knowledge_to_text(knowledge: Dict[str, Any]) -> str:
    """
    Преобразует knowledge-словарь в текст.
    Пример:
        {"country": "Poland"} -> "country=Poland"
    Если knowledge пустой -> ""
    """
    if not knowledge:
        return ""

    lines = []
    for key, value in knowledge.items():
        if value is None:
            value = ""
        lines.append(f"{key}={value}")
    return "\n".join(lines)


def dialog_to_text(dialog: List[Dict[str, str]]) -> str:
    """
    Поддерживает пока обычный список сообщений.
    В твоем текущем сценарии обычно будет один user-message,
    но функция умеет и несколько сообщений подряд.
    """
    parts = []

    for msg in dialog:
        role = msg["role"].strip().lower()
        content = msg["content"]

        if role == "user":
            role_token = SPECIAL_TOKENS["user"]
        elif role == "assistant":
            role_token = SPECIAL_TOKENS["assistant"]
        else:
            raise ValueError(f"Unsupported role: {role}")

        parts.append(role_token)
        parts.append("\n")
        parts.append(content)
        parts.append("\n")
        parts.append(SPECIAL_TOKENS["turn"])
        parts.append("\n")

    return "".join(parts)


def build_prefix_and_target(sample: Dict[str, Any]) -> (str, str):
    """
    Возвращает:
      prefix_text  - то, что идет во вход и маскируется -100
      target_text  - то, что модель должна предсказать

    Схема:
      <knowledge> in <turn>
      <dialog...>           # обычно user
      <knowledge> out <turn>
      <assistant> ... <turn>
    """
    knowledge_in_text = knowledge_to_text(sample.get("knowledge_in", {}))
    knowledge_out_text = knowledge_to_text(sample.get("knowledge_out", {}))
    dialog_text = dialog_to_text(sample["dialog"])
    assistant_text = sample["assistant"]

    prefix_parts = [
        SPECIAL_TOKENS["knowledge"], "\n",
        knowledge_in_text, "\n",
        SPECIAL_TOKENS["turn"], "\n",
        dialog_text,
    ]

    target_parts = [
        SPECIAL_TOKENS["knowledge"], "\n",
        knowledge_out_text, "\n",
        SPECIAL_TOKENS["turn"], "\n",
        SPECIAL_TOKENS["assistant"], "\n",
        assistant_text, "\n",
        SPECIAL_TOKENS["turn"],
    ]

    prefix_text = "".join(prefix_parts)
    target_text = "".join(target_parts)

    return prefix_text, target_text


def encode_sample(
    sample: Dict[str, Any],
    tokenizer,
    add_eos: bool = False,
    max_length: Optional[int] = None,
) -> Dict[str, List[int]]:
    """
    Кодирует один sample в input_ids / attention_mask / labels.

    labels:
      - prefix -> -100
      - target -> реальные token ids
    """
    prefix_text, target_text = build_prefix_and_target(sample)

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

    input_ids = prefix_ids + target_ids

    if add_eos and tokenizer.eos_token_id is not None:
        input_ids = input_ids + [tokenizer.eos_token_id]
        target_ids = target_ids + [tokenizer.eos_token_id]

    labels = [-100] * len(prefix_ids) + target_ids

    attention_mask = [1] * len(input_ids)

    if max_length is not None:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prefix_text": prefix_text,
        "target_text": target_text,
        "full_text": prefix_text + target_text,
    }


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
