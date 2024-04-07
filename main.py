import os
from transformers import Trainer
from mltools.utl.file_util import get_subfolders
from videomae.eval.inference import run_inference
from videomae.train.dataset import load_test_set, load_train_set, load_val_set
from videomae.train.compute import collate_fn, compute_metrics
from videomae.train.model import get_image_processor, get_model
from videomae.train.trainer import get_train_args
from videomae.train.transform import init_train_transform, init_val_transform
from videomae.eval.utl import display_gif, print_sample_info
from videomae.config import clip_duration, dataset_dir, model_ckpt, model_local, new_model_name


subset = get_subfolders(dataset_dir)
cls_name = get_subfolders(os.path.join(dataset_dir, subset[0]))
id2label = {i: cls for i, cls in enumerate(cls_name)}
label2id = {cls: i for i, cls in id2label.items()}

image_processor = get_image_processor(model_ckpt)
model = get_model(model_local, label2id, id2label)

train_transform = init_train_transform()
val_transform = init_val_transform()

train_dataset = load_train_set(dataset_dir, clip_duration, train_transform)
val_dataset = load_val_set(dataset_dir, clip_duration, val_transform)
test_dataset = load_test_set(dataset_dir, clip_duration, val_transform)

print("Number of video: ", train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)

sample_video = next(iter(train_dataset))
print("Key of a sample data", sample_video.keys())
print_sample_info(sample_video, id2label)

train_args = get_train_args(train_dataset)
trainer = Trainer(
    model,
    train_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

trainer.train()
trainer.evaluate(test_dataset)

trainer.save_model(new_model_name)
test_results = trainer.evaluate(test_dataset)
trainer.log_metrics("test", test_results)
trainer.save_metrics("test", test_results)
trainer.save_state()

trainer.push_to_hub()
print(f"trained model {new_model_name}")

trained_model = get_model(new_model_name, label2id, id2label)
sample_test_video = next(iter(test_dataset))
print_sample_info(sample_test_video)

logits = run_inference(trained_model, sample_test_video)
display_gif(sample_test_video["video"])

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
