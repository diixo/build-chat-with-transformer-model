
from transformers import AutoModelForCausalLM, TrainingArguments, GPT2TokenizerFast, Trainer, GenerationConfig

from text_dataset import TextDataset, collate_batch, format_prompt

import torch
import random
import numpy as np
from utils import check_local_model_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "gpt2"
LEARNING_RATE = 5e-5
EPOCHS = 20
BATCH_SIZE = 8
MAX_LENGTH = 1024

model_dir = "trained_model"
model_output_dir = model_dir


if __name__ == "__main__":

    exist, msg = check_local_model_dir(f"{model_output_dir}")

    if not exist:

        tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME, local_files_only=False, padding_side="right")

        special_tokens = {
            "pad_token": "<|pad|>",
            "additional_special_tokens": [
                "<|system|>",
                "<|user|>",
                "<|assistant|>",
                "<|knowledge|>",
            ]
        }

        num_added = tokenizer.add_special_tokens(special_tokens)

        print("added:", num_added)
        print(f"vocab size={len(tokenizer)}, pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

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

        #################################################################################

        train_dataset = TextDataset(["data/household_definitions_v6_tails.txt"], tokenizer)

        print("dataset size=", len(train_dataset))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=lambda x: collate_batch(
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

    input_prompt = "Define: table"
    prompt = format_prompt(input_prompt)

    print(f"Tokens: {tokenizer.tokenize(prompt)}")

    input_ids = tokenizer(prompt, truncation=True, add_special_tokens=False, max_length=MAX_LENGTH, return_tensors="pt")

    input_ids = input_ids["input_ids"].to(device)
    gen_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )[0]
    
    # extract answer with first token prefix: <|assistant|>
    gen_ids = gen_ids[len(input_ids[0])-1:]

    # check assistant role:
    if gen_ids[0] == tokenizer.convert_tokens_to_ids("<|assistant|>"):

        # decode with skip special tokens like eos, roles also.
        output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        print(80 * "-")
        print("### User:", input_prompt)
        print("### Assistant:", output_text.strip())
