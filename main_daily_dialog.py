
from transformers import AutoModelForCausalLM, TrainingArguments, GPT2TokenizerFast, Trainer, GenerationConfig

from dialog_loader import DialogLoader, DialogConfig, collate_fn_batch, read_jsonl_dataset

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


model_dir = "outputs/trained_daily_dialog"
model_output_dir = model_dir



def chatting(query: str, is_first_query: bool, model, tokenizer, query_cache, config, device):

    def _preprocess(query, query_cache):

        if query_cache is None:
            input_ids = [tokenizer.cls_token_id] + tokenizer.encode(query) + [tokenizer.sep_token_id]
            query_cache = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        else:
            input_ids = tokenizer.encode(query) + [tokenizer.sep_token_id]
            new_tokens = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
            query_cache = torch.cat([query_cache, new_tokens], dim=1)
        return query_cache

    query_cache = None if is_first_query else query_cache
    query_cache = _preprocess(query, query_cache)

    input_len = query_cache.size(1)
    max_new_tokens = config.max_len - input_len

    if max_new_tokens <= 0:
        return None, "", True, True

    with torch.no_grad():
        generated = model.generate(
            input_ids=query_cache,
            attention_mask = torch.ones_like(query_cache, device=query_cache.device)
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=[tokenizer.sep_token_id, tokenizer.eos_token_id],
            pad_token_id=tokenizer.eos_token_id,
        )

    full_sequence = generated[0]
    answer_tokens = full_sequence[input_len:].tolist()

    query_done = False
    if answer_tokens:
        if answer_tokens[-1] == tokenizer.eos_token_id:
            query_done = True

    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    # query_done can use to skip current topic
    if query_done:
        query_cache = None
        is_first_query = True
        print("query_cache.sz: empty")
    else:
        query_cache = full_sequence.unsqueeze(0)
        is_first_query = False
        print("query_cache.sz:", query_cache.shape)

    return query_cache, answer, query_done, is_first_query



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
            "cls_token": config.token_cls,
            "sep_token": config.token_sep,
            "pad_token": config.token_pad,
            "additional_special_tokens": []
        }

        num_added = tokenizer.add_special_tokens(special_tokens)

        print("added:", num_added)
        print(f"vocab size={len(tokenizer)}, pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

        ###################################################################################################################

        train_dataset = DialogLoader(
            read_jsonl_dataset("data/daily_dialog/daily-dialog_all.jsonl"),
            tokenizer,
            config)


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
            data_collator=lambda x: collate_fn_batch(
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

    query_cache = None
    is_first_query = True

    while 1:
        query = input("Q: ")
        if query == 'exit':
            break

        query_cache, answer, query_done, is_first_query = chatting(
            query,
            is_first_query,
            model,
            tokenizer,
            query_cache,
            config,
            device)

        print("A:", answer)
