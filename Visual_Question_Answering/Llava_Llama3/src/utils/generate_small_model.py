import torch
import transformers
import accelerate


def generate_small_llama_model(source_model_id, save_path):
    config = transformers.AutoConfig.from_pretrained(
        source_model_id,
        trust_remote_code=True,
    )
    config._name_or_path = source_model_id
    config.hidden_size = 4
    config.intermediate_size = 14
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.num_hidden_layers = 2
    config.torch_dtype = "bfloat16"

    model = transformers.AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
    )

    with accelerate.init_empty_weights():
        model.generation_config = transformers.AutoModelForCausalLM.from_pretrained(source_model_id).generation_config

    model = model.to(torch.bfloat16)
    model.save_pretrained(save_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        source_model_id,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(save_path)
