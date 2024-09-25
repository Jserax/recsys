from typing import Any, Dict, Tuple

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base="1.3", config_path="./config", config_name="train_mf_implicit.yaml"
)
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    wandb.login()
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    callbacks = hydra.utils.instantiate(cfg.callbacks)
    logger = hydra.utils.instantiate(cfg.logger)
    trainer = hydra.utils.instantiate(cfg.trainer)
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    trainer.fit(model=model, datamodule=datamodule)
    train_metrics = trainer.callback_metrics
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}
    trainer.logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    wandb.finish()
    return metric_dict, object_dict


if __name__ == "__main__":
    train()
