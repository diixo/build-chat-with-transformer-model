
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

HISTORY_NUM_PAIRS = 2


def chatting(query: str, model, tokenizer, query_cache, config, device):
    """
    query_cache: None или list[list[int]]

    Хранит историю как список реплик:
        [user1, assistant1, user2, assistant2]

    Максимум 4 реплики.
    При добавлении новой пары, если длина > 4,
    удаляется самая старая пара (FIFO).
    
    Семантика:
    - SEP -> обычное завершение ответа, историю сохраняем
    - EOS -> query_done=True, историю очищаем как гипотетический конец топика
    """

    def build_prompt_from_history(history):
        flat_tokens = []
        for turn_tokens in history:
            flat_tokens.extend(turn_tokens)
        return torch.tensor(flat_tokens, dtype=torch.long, device=device).unsqueeze(0)

    if query_cache is None:
        query_cache = []

    # 1. attach user-utterance
    user_tokens = [tokenizer.cls_token_id] + tokenizer.encode(query) + [tokenizer.sep_token_id]
    query_cache.append(user_tokens)

    # 2. build prompt from history
    input_ids = build_prompt_from_history(query_cache)

    input_len = input_ids.size(1)
    max_new_tokens = config.max_len - input_len

    if max_new_tokens <= 0:
        return None, "", True

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            #attention_mask=torch.ones_like(input_ids, device=input_ids.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=[tokenizer.sep_token_id, tokenizer.eos_token_id],
            pad_token_id=tokenizer.eos_token_id,
        )

    full_sequence = generated[0]
    answer_tokens = full_sequence[input_len:].tolist()

    # query_done=True if model completed answe by eos
    query_done = False
    if answer_tokens:
        if answer_tokens[-1] == tokenizer.eos_token_id:
            query_done = True

    # remove final SEP/EOS from the answer
    if answer_tokens and answer_tokens[-1] in [tokenizer.sep_token_id, tokenizer.eos_token_id]:
        answer_tokens = answer_tokens[:-1]

    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    # 3. If the topic is not completed, save the answer to history
    if not query_done:
        assistant_tokens = answer_tokens + [tokenizer.sep_token_id]
        query_cache.append(assistant_tokens)

        # FIFO by pairs: maximum is HISTORY_NUM_PAIRS utterance pairs 
        if len(query_cache) > (2*HISTORY_NUM_PAIRS):
            query_cache = query_cache[2:]
    else:
        # clear cache as marker of reset current topic
        query_cache = None
        print("INFO: query_cache=0")

    return query_cache, answer, query_done



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

    print(f"EOS_id: {tokenizer.eos_token_id}, PAD_id: {tokenizer.pad_token_id} SEP_id: {tokenizer.sep_token_id}, ")

    query_cache = None

    while 1:
        query = input("Q: ")
        if query == 'exit':
            break

        query_cache, answer, query_done = chatting(
            query,
            model,
            tokenizer,
            query_cache,
            config,
            device
        )

        print("A:", answer)
