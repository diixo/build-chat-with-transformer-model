
from transformers import AutoModelForCausalLM, TrainingArguments, GPT2TokenizerFast, Trainer

from dialog_dataset import DialogDataset, collate_lm_batch
from torch.utils.data import ConcatDataset

import torch
import random
import numpy as np
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
EPOCHS = 5
BATCH_SIZE = 8
MAX_LENGTH = 1024

model_dir = "trained_model"
model_output_dir = model_dir


if __name__ == "__main__":

    tokenizer = GPT2TokenizerFast.from_pretrained(
        MODEL_NAME,
        additional_special_tokens=["User:", "Assistant:"],
        padding_side="right",
        model_max_length=MAX_LENGTH
        )

    train_dataset = ConcatDataset([
        DialogDataset("data/test.txt", tokenizer),
        DialogDataset("data/dialogue_datset_300.txt", tokenizer),
        DialogDataset("data/dialogue_datset_700.txt", tokenizer),
        DialogDataset("data/dialogue_datset_2000.txt", tokenizer),
        DialogDataset("data/dialogue_datset_2000_v2.txt", tokenizer),
        DialogDataset("data/dialogue_datset_5000.txt", tokenizer),
    ])


    print(len(train_dataset))

    for item in train_dataset:
        print(tokenizer.decode(item["input_ids"]))
        print(item["labels"])
        break
    
    exit(0)
    ##################################################################

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)

    training_args = TrainingArguments(
        output_dir=model_dir,
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
            padding_value=tokenizer.eos_token_id,
            label_padding_value=tokenizer.eos_token_id
        ),
    )

    trainer.train()
    #trainer.save_model(model_dir)

    #model.save_pretrained(model_output_dir)
    #tokenizer.save_pretrained(model_output_dir)
