
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


def chatting(query: str, is_first_query, model, tokenizer, query_cache):

    def _preprocess(query, is_first_query, query_cache=None):
        if is_first_query:
            query = [tokenizer.cls_token_id] + tokenizer.encode(query) + [tokenizer.sep_token_id]
            query_cache = torch.tensor(query, dtype=torch.long).unsqueeze(0).to(device)
        else:
            query = tokenizer.encode(query) + [tokenizer.sep_token_id]
            query_cache = torch.cat([query_cache, torch.tensor(query, dtype=torch.long).unsqueeze(0).to(device)], dim=1)
        return query_cache

    query_cache = None if is_first_query else query_cache
    query_cache = _preprocess(query, is_first_query, query_cache)
    query_done = False
    is_first_query = False

    answer = []
    while 1:
        output = model(query_cache)
        output = output.logits
        pred_token = torch.argmax(output[:, -1], dim=-1)
        answer.append(pred_token.item())
        query_cache = torch.cat((query_cache, pred_token.unsqueeze(1)), dim=1)

        if pred_token == tokenizer.sep_token_id:
            answer.pop()
            break
        elif pred_token == tokenizer.eos_token_id:
            answer.pop()
            query_done = True
            break

        if query_cache.size(1) >= config.max_len:
            query_done = True
            break

        if query_done:
            query_cache = None
            is_first_query = True

    answer = tokenizer.decode(answer)
    return query_cache, answer, query_done, is_first_query


# def chatting(model, tokenizer):

#     turn_token = config.token_sep

#     print("Type 'exit' to stop.\n")

#     history = ""

#     while True:

#         user_msg = input("### User: ").strip()
#         if user_msg.lower() in {"exit", "quit"}:
#             break

#         #prompt = history + f"User: {user_msg}\n{assistant}:"

#         prompt = f"{user_msg} {turn_token}"

#         input_ids = tokenizer(prompt, truncation=True, add_special_tokens=False, max_length=MAX_LENGTH, return_tensors="pt")

#         prompt_len = input_ids["input_ids"].shape[1]

#         input_ids = input_ids["input_ids"].to(device)
#         gen_ids = model.generate(
#                 input_ids=input_ids,
#                 max_new_tokens=50,
#                 do_sample=False,
#                 eos_token_id=tokenizer.eos_token_id,
#                 pad_token_id=tokenizer.pad_token_id
#             )[0]

#         gen_ids = gen_ids[prompt_len : ]

#         answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

#         print(f"### Assistant: {answer}")

#         history += f"### User: {user_msg}\n: {answer}"



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
        query = input('Q: ')
        if query == 'exit':
            break

        query, answer, query_done, is_first_query = chatting(
            query,
            is_first_query,
            model,
            tokenizer,
            query_cache)
