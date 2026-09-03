from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    num_pos_tokens: int
    path: str
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    sep_token_id: int = 3
    nil_token_id: int = 4
    pos_token_offset: int = 5
    max_seq_length: int = 8192
    attn_implementation: str = "flash_attention_2"
    # Architecture overrides (hidden_size / num_hidden_layers / ...). 
    # When set, the model is built from config rather than from pretrained weights
    overrides: dict = None
    from_scratch: bool = False
    # Relative-v_t variant: emit (v_t - v_s) in a dedicated band of
    # 2*num_pos_tokens-1 ids starting at delta_token_offset. 
    relative_vt: bool = False
    delta_token_offset: int = 0


@dataclass
class DataConfig:
    path: str
    num_pos_tokens: int
    # When True, expect ``seed_mesh_q`` in the dataset and prepend it at train /
    # eval time (mask prefix + base soup). When False, run upstream's
    # unconditional layout and require the column to be absent.
    use_seed_mesh: bool = False
    pad_to_multiple_of: int = 32
    padding_side: str = "right"
    random_scale: bool = True
    random_scale_max: float = 1.1
    random_shift: bool = True
    random_shift_min: float = -0.1
    random_shift_max: float = 0.1
    random_rotate: bool = True
    overfit: bool = False
