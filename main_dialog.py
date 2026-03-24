
from transformers import AutoModelForCausalLM, TrainingArguments, GPT2TokenizerFast, Trainer, GenerationConfig

from dialog_dataset import DialogDataset, DialogConfig, collate_lm_batch

import torch
import random
import numpy as np
from utils import check_local_model

from transformers import set_seed


seed = 42
set_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# disabled TF32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_NAME = "gpt2"
LEARNING_RATE = 1e-4
EPOCHS = 30
BATCH_SIZE = 8
MAX_LENGTH = 1024

config = DialogConfig()

model_dir = "outputs/trained_model_dialog"
model_output_dir = model_dir


def chatting(model, tokenizer, turn_token=None):

    if turn_token is None:
        turn_token_id = tokenizer.eos_token_id
    else:
        turn_token_id = tokenizer.convert_tokens_to_ids(turn_token)

    print("Type 'exit' to stop.\n")

    history = ""

    while True:

        user_msg = input("### User: ").strip()
        if user_msg.lower() in {"exit", "quit"}:
            break

        #prompt = history + f"User: {user_msg}\n{assistant}:"

        prompt = f"<|user|> {user_msg} <|turn|>\n<|assistant|>"

        input_ids = tokenizer(prompt, truncation=True, add_special_tokens=False, max_length=MAX_LENGTH, return_tensors="pt")

        prompt_len = input_ids["input_ids"].shape[1]

        input_ids = input_ids["input_ids"].to(device)
        gen_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                do_sample=False,
                eos_token_id=[tokenizer.eos_token_id, turn_token_id],
                pad_token_id=tokenizer.pad_token_id
            )[0]

        gen_ids = gen_ids[prompt_len : ]

        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        print(f"### Assistant: {answer}")

        history += f"### User: {user_msg}\nAssistant: {answer}"



if __name__ == "__main__":

    exist, msg = check_local_model(f"{model_output_dir}")

    if not exist:
        tokenizer = GPT2TokenizerFast.from_pretrained(
            MODEL_NAME,
            local_files_only=False,
            padding_side="right",
            model_max_length=MAX_LENGTH
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

        print("added:", num_added)
        print(f"vocab size={len(tokenizer)}, pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

        ###################################################################################################################

        train_dataset = DialogDataset([
            "data/dialogues_clarification_64.txt",
            # "data/dialogues_clarification_12000.txt",
            # "data/dialogues_weather.txt",
        ], tokenizer, config)


        print(f"dialogues: size={len(train_dataset)}")
        #train_dataset.save_to_jsonl("dialogs_clarification-12k.jsonl")

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
                padding_id=tokenizer.pad_token_id,
                label_padding_id=-100
            ),
        )

        trainer.train()
        trainer.save_model(model_output_dir)

        #model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
    else:

        tokenizer = GPT2TokenizerFast.from_pretrained(model_output_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_output_dir, local_files_only=True).to(device)


    #############################################################################

    print("EOS_id:", tokenizer.eos_token_id, ", BOS_id:", tokenizer.bos_token_id, ", PAD_id:", tokenizer.pad_token_id)


    chatting(model, tokenizer)
