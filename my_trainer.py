from transformers import Trainer
import torch.nn as nn

class MyTrainer(Trainer):

    IGNORE_INDEX = -100

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        x = inputs["input_ids"]
        y = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)

        logits = model(x)
        # or model(x, attention_mask=attention_mask)

        loss_fn = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        loss = loss_fn(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            y[:, 1:].reshape(-1)
        )

        return (loss, {"logits": logits}) if return_outputs else loss
