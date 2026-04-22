from bonsai.gemma3.modeling import Gemma3TextScaledWordEmbedding, TextConfig, AttentionMode, TextShardConfig

import jax.numpy as jnp
from flax import nnx
import json
import os
from huggingface_hub import snapshot_download
from safetensors import safe_open
from flax import nnx

# in the file directory, there should be a folder for bonsai
def load_model():
    print("Loading model...")
    model_name = "google/embeddinggemma-300m"
    access_token = os.environ["HF_TOKEN"]
    # downloads the model weights and the configuration file
    ckpt_path = snapshot_download(model_name, token=access_token)
    config_path = os.path.join(ckpt_path, "config.json")
    safetensor_path = os.path.join(ckpt_path, "model.safetensors")
    weights = {}
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    # loads the model config from https://huggingface.co/google/embeddinggemma-300m/blob/main/config.json
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    # No sharding needed for single-GPU inference
    shd_cfg = TextShardConfig().no_sharding()
    layer_types = []
    for type in cfg_dict["layer_types"]:
        if (type == "sliding_attention"):
            layer = AttentionMode.SLIDE
        else:
            layer = AttentionMode.FULL
        layer_types.append(layer)
    text_cfg = TextConfig(
        vocab_size=cfg_dict["vocab_size"],
        hidden_size=cfg_dict["hidden_size"],
        head_dim=cfg_dict["head_dim"],
        intermediate_size=cfg_dict["intermediate_size"],
        layer_types=layer_types,
        attention_dropout=cfg_dict["attention_dropout"],
        attention_bias=cfg_dict["attention_bias"],
        max_position_embeddings=cfg_dict["max_position_embeddings"],
        num_attention_heads=cfg_dict["num_attention_heads"],
        num_hidden_layers=cfg_dict["num_hidden_layers"],
        num_key_value_heads=cfg_dict["num_key_value_heads"],
        norm_dtype=jnp.float32,
        sliding_window=cfg_dict["sliding_window"],
        rms_norm_eps=cfg_dict["rms_norm_eps"],
        rope_full_factor=1.0, 
        rope_full_theta=1000000.0,
        rope_slide_factor=1.0,
        rope_slide_theta=10000.0,
        shd_cfg=shd_cfg
    )
    # RNG seed 0 is fine for deterministic inference — no dropout or sampling
    rngs = nnx.Rngs(0)
    model = Gemma3TextScaledWordEmbedding(cfg=text_cfg, rngs=rngs)
    embedding_key = "embed_tokens.weight"
    pretrained_jax_weight = jnp.array(weights[embedding_key])
    model.weight.embedding = pretrained_jax_weight
    return model
