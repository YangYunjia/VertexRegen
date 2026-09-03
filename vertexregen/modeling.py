from transformers import OPTConfig, OPTForCausalLM
from .config import ModelConfig


class VertexRegenConfig(OPTConfig):
    model_type = "vertexregen"

    def __init__(
        self, num_pos_tokens=128, nil_token_id=4, pos_token_offset=5,
        relative_vt=False, delta_token_offset=0, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_pos_tokens = num_pos_tokens
        self.nil_token_id = nil_token_id
        self.pos_token_offset = pos_token_offset
        # Carried on the HF config, not just ModelConfig, because SequenceDecoder
        # is constructed from the saved model config at generation time.
        self.relative_vt = relative_vt
        self.delta_token_offset = delta_token_offset


class VertexRegenModel(OPTForCausalLM):
    config_class = VertexRegenConfig

    def __init__(self, config: VertexRegenConfig):
        super().__init__(config)


def get_model(model_cfg: ModelConfig):
    config = VertexRegenConfig.from_pretrained(
        model_cfg.path,
        vocab_size=model_cfg.vocab_size,
        num_pos_tokens=model_cfg.num_pos_tokens,
        pad_token_id=model_cfg.pad_token_id,
        bos_token_id=model_cfg.bos_token_id,
        eos_token_id=model_cfg.eos_token_id,
        sep_token_id=model_cfg.sep_token_id,
        nil_token_id=model_cfg.nil_token_id,
        pos_token_offset=model_cfg.pos_token_offset,
        relative_vt=getattr(model_cfg, "relative_vt", False),
        delta_token_offset=getattr(model_cfg, "delta_token_offset", 0),
        max_position_embeddings=model_cfg.max_seq_length,
        do_layer_norm_before=True,
    )
    for k, v in (model_cfg.overrides or {}).items():
        setattr(config, k, v)
    config.word_embed_proj_dim = config.hidden_size
    if model_cfg.overrides or model_cfg.from_scratch:
        # Capacity was changed to match the other arms, so pretrained weights no
        # longer line up; build from config instead of silently random-initialising
        # most of the checkpoint via ignore_mismatched_sizes.
        model = VertexRegenModel(config)
    else:
        model = VertexRegenModel.from_pretrained(
            model_cfg.path, config=config, ignore_mismatched_sizes=True
        )
    model.set_attn_implementation(model_cfg.attn_implementation)
    n = sum(p.numel() for p in model.parameters())
    print(f"[vertexregen] hidden={config.hidden_size} layers={config.num_hidden_layers} "
          f"heads={config.num_attention_heads} ffn={config.ffn_dim} -> {n/1e6:.2f}M params",
          flush=True)
    return model
