
from transformers import AutoModelForCausalLM, TrainingArguments, GPT2TokenizerFast, Trainer

from dialog_dataset import DialogDataset, make_lm_collate_fn


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
BATCH_SIZE = 4

model_dir = "trained_model"


if __name__ == "__main__":


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

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)

    train_dataset = DialogDataset("data/test.txt", tokenizer)

    print(len(train_dataset))


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_lm_collate_fn(tokenizer),
    )

    #trainer.train()
    #trainer.save_model(model_dir)
