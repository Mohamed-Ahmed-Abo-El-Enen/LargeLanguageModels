import os
from pathlib import Path


def re_direct_hf_cache(HF_HOME, HF_DATASETS_CACHE, TRANSFORMERS_CACHE):
    Path(HF_HOME).mkdir(parents=True, exist_ok=True)
    Path(HF_DATASETS_CACHE).mkdir(parents=True, exist_ok=True)
    Path(TRANSFORMERS_CACHE).mkdir(parents=True, exist_ok=True)

    os.environ['HF_HOME'] = HF_HOME
    os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE
    os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE
