from transformers import TrainingArguments
from videomae.config import batch_size, new_model_name, num_epochs


def get_train_args(train_dataset) -> TrainingArguments:
    return TrainingArguments(
        new_model_name,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
    )
