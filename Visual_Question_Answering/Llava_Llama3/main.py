from src.utils.hf_utils import re_direct_hf_cache
import wandb
from huggingface_hub.hf_api import HfFolder
from src.trainer_utils.train_arguments import *
from src.train import *

if __name__ == "__main__":
    HF_HOME = 'transformers_cache/huggingface'
    HF_DATASETS_CACHE = 'transformers_cache/huggingface/datasets'
    TRANSFORMERS_CACHE = 'transformers_cache/huggingface/models'
    re_direct_hf_cache(HF_HOME, HF_DATASETS_CACHE, TRANSFORMERS_CACHE)

    wandb.login(key="YOUR WANDB KEY")
    HfFolder.save_token('YOUR HUGGING FACE KEY')

    output_dir = "../checkpoint/llava-llama-3-8b-qlora"

    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = TrainingArguments(output_dir)


    model_args.model_name_or_path = "transformers_cache/huggingface/hub/CustomLlama3"
    model_args.vision_tower = "transformers_cache/huggingface/hub/openai-clip-vit-base-patch16"
    data_args.data_path = "/dataset/train/dataset.json"
    data_args.image_folder = "/dataset/images/"
    data_args.validation_data_path = "dataset/validation/dataset.json"
    training_args.output_dir = output_dir
    training_args.lora_enable = True
    training_args.lora_r = 128
    training_args.lora_alpha = 256
    training_args.mm_projector_lr = 2e-5
    training_args.bits = 4
    model_args.version = "llama3"
    model_args.mm_projector_type = "mlp2x_gelu"
    model_args.mm_vision_select_layer = -2
    model_args.mm_use_im_start_end = False
    model_args.mm_use_im_patch_token = False
    data_args.image_aspect_ratio = "pad"
    training_args.group_by_modality_length = True
    training_args.bf16 = False
    training_args.fp16 = False
    # training_args.num_train_epochs = 500
    training_args.max_steps = 10
    training_args.per_device_train_batch_size = 1
    training_args.per_device_eval_batch_size = 1
    training_args.gradient_accumulation_steps = 1
    training_args.evaluation_strategy = "epoch"
    training_args.save_strategy = "steps"
    training_args.save_steps = 1
    training_args.save_total_limit = 1
    training_args.learning_rate = 2e-4
    training_args.weight_decay = 0.
    training_args.warmup_ratio = 0.03
    training_args.lr_scheduler_type = "cosine"
    training_args.logging_steps = 1
    training_args.tf32 = False
    training_args.model_max_length = 512
    training_args.gradient_checkpointing = True
    training_args.dataloader_num_workers = 2
    data_args.lazy_preprocess = True
    # training_args.report_to = "wandb"
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    train(model_args, data_args, training_args)