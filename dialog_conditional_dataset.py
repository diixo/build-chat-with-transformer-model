
from typing import Dict, List, Any, Optional, Tuple
import torch

import json
from torch.utils.data import Dataset
from dialog_dataset import DialogConfig
from transformers import GPT2TokenizerFast, Trainer, AutoModelForCausalLM, TrainingArguments
from utils import check_local_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = DialogConfig()


def chatting(model, tokenizer, turn_token=None):

    if turn_token is None:
        turn_token_id = tokenizer.eos_token_id
    else:
        turn_token_id = tokenizer.convert_tokens_to_ids(turn_token)

    knowledge_token_id = tokenizer.convert_tokens_to_ids(config.token_knowledge)

    print("Type 'exit' to stop.\n")

    while True:

        user_msg = input("### User: ").strip()
        if user_msg.lower() in {"exit", "quit"}:
            break


        prompt = f"<|user|> {user_msg} <|turn|>\n<|assistant|>"

        input_ids = tokenizer(prompt, truncation=True, add_special_tokens=False, max_length=config.max_length, return_tensors="pt")

        prompt_len = input_ids["input_ids"].shape[1]

        input_ids = input_ids["input_ids"].to(device)
        gen_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                do_sample=False,
                eos_token_id=[tokenizer.eos_token_id],
                pad_token_id=tokenizer.pad_token_id
            )[0]

        gen_ids = gen_ids[prompt_len : ]

        knowledge_pos = (gen_ids == knowledge_token_id).nonzero()

        if knowledge_pos.numel() > 0:
            split_idx = knowledge_pos[0].item()
            answer_ids = gen_ids[:split_idx]
            knowledge_ids = gen_ids[split_idx + 1:]
        else:
            answer_ids = gen_ids
            knowledge_ids = None

        answer_full = tokenizer.decode(gen_ids, skip_special_tokens=False)

        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

        knowledge = "" if knowledge_ids is None else tokenizer.decode(knowledge_ids, skip_special_tokens=False)

        print(f"### Assistant: {answer.strip()}")



class DialogConditionDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: Optional[int] = None,
        add_eos: bool = True,
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


    def _validate_dialog(self, dialog: List[Dict[str, str]]) -> None:
        if not dialog:
            raise ValueError("dialog must not be empty")

        prev_role = None
        for i, msg in enumerate(dialog):

            role = msg["role"].strip().lower()
            if role not in ("user", "assistant"):
                raise ValueError(f"Unsupported role: {role}")

            if prev_role == role:
                raise ValueError(
                    f"Roles must alternate, but got two '{role}' in a row at index {i}"
                )
            prev_role = role

        last_role = dialog[-1]["role"].strip().lower()
        if last_role != "assistant":
            raise ValueError("The last dialog message must be assistant")


    def _tokenize(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _knowledge_to_text(self, knowledge: Dict[str, Any]) -> str:
        if not knowledge:
            return ""
        lines = []
        for k, v in knowledge.items():
            if v is None:
                v = ""
            lines.append(f"{k}={v}")
        return "\n".join(lines)


    def _build_knowledge_block(self, knowledge: Dict[str, Any]) -> str:
        body = self._knowledge_to_text(knowledge)
        if body:
            return f"{self.tok_knowledge}\n{body}\n{self.tok_turn}"
        else:
            return f"{self.tok_knowledge}{self.tok_turn}"


    def _encode_sample(self, sample: Dict[str, Any]) -> Dict[str, List[int]]:
        knowledge_in_text = self._build_knowledge_block(sample.get("knowledge_in", {}))
        knowledge_out_text = self._build_knowledge_block(sample.get("knowledge_out", {}))

        input_ids: List[int] = []
        labels: List[int] = []

        # knowledge_in: always context only
        k_in_ids = self._tokenize(knowledge_in_text)
        input_ids.extend(k_in_ids)
        labels.extend([-100] * len(k_in_ids))

        # dialog
        dialog = sample["dialog"]
        self._validate_dialog(dialog)

        for msg in dialog:
            role = msg["role"].strip().lower()
            content = msg["content"]


            if role == "user":
                train_this_block = False
                block = f"{self.tok_user} {content} {self.tok_turn}\n{self.tok_assistant}"
            else:
                train_this_block = True
                block = f"{content} {self.tok_turn}\n"

            block_ids = self._tokenize(block)
            input_ids.extend(block_ids)

            if train_this_block:
                labels.extend(block_ids)
            else:
                labels.extend([-100] * len(block_ids))

        # knowledge_out: train
        k_out_ids = self._tokenize(knowledge_out_text)
        input_ids.extend(k_out_ids)
        labels.extend(k_out_ids)

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

    MODEL_NAME = "gpt2"

    model_dir = "outputs/trained_conditional_dialog"
    model_output_dir = model_dir

    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 30

    exist, msg = check_local_model(f"{model_output_dir}")

    if not exist:

        tokenizer = GPT2TokenizerFast.from_pretrained(
            "gpt2",
            local_files_only=False,
            padding_side="right",
            model_max_length=1024
            )

        special_tokens = {
            "pad_token": "<|pad|>",
            "additional_special_tokens": [
                config.token_user,
                config.token_assistant,
                config.token_knowledge,
                config.token_turn,
            ]
        }

        num_added = tokenizer.add_special_tokens(special_tokens)

        print(f"added new: {num_added}, vocab sz={len(tokenizer)}, pad_id={tokenizer.pad_token_id}")


        train_dataset = DialogConditionDataset(
            file_path="data/condition-dataset-7-slots.json",
            tokenizer=tokenizer,
            max_length=256,
            add_eos=False,
        )

        print(f"input dataset.sz={len(train_dataset)}")


        item = train_dataset[0]
        # print(item.keys())
        # print(len(item["input_ids"]))
        # print(len(item["labels"]))

        ##################################################################

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=False)

        # resize token embeddings without mean recalculation
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        model.config.pad_token_id = tokenizer.pad_token_id

        model.to(device)

        training_args = TrainingArguments(
            output_dir=model_output_dir,
            save_strategy="no",
            eval_strategy="no",
            learning_rate=LEARNING_RATE,
            num_train_epochs=EPOCHS,
            weight_decay=0.0,
            push_to_hub=False,
            load_best_model_at_end=False,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            lr_scheduler_type="constant",

            bf16=True,
            fp16=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=lambda x: collate_lm_batch(
                x,
                padding_value=tokenizer.pad_token_id,
                label_padding_value=-100
            ),
        )

        trainer.train()
        trainer.save_model(model_output_dir)

        #model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
    else:

        tokenizer = GPT2TokenizerFast.from_pretrained(model_output_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_output_dir, local_files_only=True).to(device)

    chatting(model, tokenizer, turn_token=config.token_turn)
    #chatting(model, tokenizer)
