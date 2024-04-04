# For base videoMAE
from pathlib import Path


dataset_dir = "G:\\CODE\\VIDEOMAE\\videomae\\DATA\VIDEOMAE\\bekhoaxe\\splited_videos"
model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"  # pre-trained model from which to fine-tune
model_local = Path("G:\\CODE\\VIDEOMAE\\videomae\\DATA\\VIDEOMAE\\model\\videomae-base-finetuned-kinetics")
batch_size = 2  # batch size for training and evaluation

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
input_size = (224, 224)

sample_rate = 4
fps = 30
num_frames_to_sample = 16
clip_duration = num_frames_to_sample * sample_rate / fps

model_name = "videomae-base"
new_model_name = f"{model_name}-finetuned-bekhoaxe"
num_epochs = 4
