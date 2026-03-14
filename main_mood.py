
from transformers import AutoModelForCausalLM, TrainingArguments, GPT2TokenizerFast, Trainer, GenerationConfig

from dialog_dataset import DialogDataset, collate_lm_batch

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
EPOCHS = 10
BATCH_SIZE = 8
MAX_LENGTH = 1024

model_dir = "trained_model"
model_output_dir = model_dir


def dialog(model, tokenizer, gen_cfg):

    print("Type 'exit' to stop.\n")

    history = ""

    while True:

        user_msg = input("User: ").strip()
        if user_msg.lower() in {"exit", "quit"}:
            break

        #prompt = history + f"User: {user_msg}\n{assistant}:"

        prompt = f"User: {user_msg}\nAssistant:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, generation_config=gen_cfg)

        prompt_len = inputs["input_ids"].shape[1]

        out = out[0][prompt_len:]   # cut out the prompt tokens, keep only the generated part

        full = tokenizer.decode(out, skip_special_tokens=True)

        # cut out the answer from the full generated text — отрезаем всё до последнего "Lexor:", а дальше — до следующего "User:" (если есть)
        tail = full.split("Assistant:")[-1]

        # часто модель тянет дальше "User:" — обрежем
        answer = tail.split("User:")[0].strip()

        print(f"Assistant: {answer}\n")

        history += f"User: {user_msg}\nAssistant: {answer}\n"



if __name__ == "__main__":

    ok, msg = check_local_model(f"{model_output_dir}")

    if not ok:
        tokenizer = GPT2TokenizerFast.from_pretrained(
            MODEL_NAME,
            local_files_only=False,
            padding_side="right",
            model_max_length=MAX_LENGTH
            )

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no pad_token_id and no eos_token_id to use as pad.")
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset = DialogDataset([
            #"data/test_50.txt",

            "data/dialogue_dataset_300.txt",
            "data/dialogue_dataset_700.txt",
            "data/dialogue_dataset_2000.txt",
            "data/dialogue_dataset_2000_v2.txt",
            "data/dialogue_dataset_5000_v3.txt",

            "data/dialogue_mood_3000.txt",
            "data/dialogue_mood_12000.txt",
            "data/dialogue_mood_20000_v3.txt",

            "data/dialogues_clarification_12000.txt",

            "data/assistant_reasoning/assistant_reasoning_dialogues_part1.txt",
            "data/assistant_reasoning/assistant_reasoning_dialogues_part2.txt",
            "data/assistant_reasoning/assistant_reasoning_dialogues_part3.txt",
        ], tokenizer)


        print("dialogues: size=", len(train_dataset))


        ##################################################################

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=False)
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

    #############################################################################

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id to use as pad.")
        tokenizer.pad_token = tokenizer.eos_token

    print("EOS_id:", tokenizer.eos_token_id, ", BOS_id:", tokenizer.bos_token_id, ", PAD_id:", tokenizer.pad_token_id)

    gen_cfg = GenerationConfig(
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )

    dialog(model, tokenizer, gen_cfg)
