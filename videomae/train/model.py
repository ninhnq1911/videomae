from typing import Any, Dict
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification


def get_model(model_ckpt: str, label2id: Dict, id2label: Dict) -> VideoMAEForVideoClassification:
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )
    return model


def get_image_processor(model_ckpt: str) -> VideoMAEImageProcessor:
    return VideoMAEImageProcessor.from_pretrained(model_ckpt)
