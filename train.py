import datasets
import hydra
import transformers
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from vertexregen.modeling import get_model
from vertexregen.config import DataConfig, ModelConfig
from transformers import EarlyStoppingCallback
from vertexregen.misc import JsonlLoggerCallback, get_last_checkpoint
from vertexregen.data import get_dataset, VertexRegenTokenizer, VertexRegenCollator

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    OmegaConf.resolve(cfg)

    accelerator = Accelerator()
    accelerator.print(OmegaConf.to_yaml(cfg))
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    data_cfg = DataConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))

    train_args = SFTConfig(
        **OmegaConf.to_container(cfg.train.train_args, resolve=True),
        max_length=model_cfg.max_seq_length,
        dataset_kwargs={
            "skip_prepare_dataset": True,
        },
        remove_unused_columns=False,
    )

    with accelerator.local_main_process_first():
        model = get_model(model_cfg)
        train_set, val_set = get_dataset(data_cfg)

    # The quantisation frame lives in the DATASET's encode_meta.json. Copy it in beside the weights.
    try:
        import os as _os, shutil as _sh
        _src = _os.path.join(str(cfg.data.path), "encode_meta.json")
        _dst = _os.path.join(str(cfg.train.train_args.output_dir), "encode_meta.json")
        _os.makedirs(_os.path.dirname(_dst), exist_ok=True)
        if _os.path.isfile(_src):
            _sh.copy2(_src, _dst); accelerator.print(f"[frame] persisted -> {_dst}")
        else:
            accelerator.print(f"[frame] WARNING: {_src} missing; outputs undecodable")
    except Exception as _e:  # noqa: BLE001
        accelerator.print(f"[frame] WARNING: could not persist encode_meta.json: {_e}")

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=VertexRegenCollator(data_cfg, model_cfg),
        processing_class=VertexRegenTokenizer(model_cfg),

        # Patience is in EVAL events, so 20 x eval_steps of no improvement.
        callbacks=[
            JsonlLoggerCallback(log_file_path=cfg.train.train_args.logging_dir),
            EarlyStoppingCallback(
                early_stopping_patience=int(
                    OmegaConf.select(cfg, "train.early_stopping_patience",
                                     default=20)),
                early_stopping_threshold=float(
                    OmegaConf.select(cfg, "train.early_stopping_threshold",
                                     default=0.0)),
            ),
        ],
    )

    trainer.train(resume_from_checkpoint=get_last_checkpoint(train_args.output_dir))
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
