from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import wandb

# for saving the item embeddings
import numpy as np

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    item_part,
)


wandb.login(key='a0c6d8a4a5a10e28e40d0086c3a2ff2103cad502')
log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # get the valid_rating_matrix and test_rating_matrix, and item_size
    datamodule.setup()
    cfg.model.net.item_size = datamodule.hparams.others.item_size
    valid_rating_matrix = datamodule.hparams.others.valid_rating_matrix
    log.info('Setup is done')

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.rating_matrix(valid_rating_matrix=valid_rating_matrix) #instantiating the rating matrix

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    print('Printing the callbacks:' , callbacks)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
    # data.others.model_type=Bert4Rec
    # log.info(f"The model is {cfg.data.others.model_type}")
    
    if cfg.get("oie_learning") and cfg.get("ckpt_path") is not None:
        log.info("Starting training using ordered item embedding!")
        if cfg.get("item"):
            log.info('Using the item embeddings')
            from models.seqrec_module import SeqRecLitModule
            checkpoint = cfg.get("ckpt_path")
            model_pt = SeqRecLitModule.load_from_checkpoint(checkpoint)
            model_dict_pt = model_pt.state_dict()
            key = 'net.item_embeddings.weight'
            embedding_dict = {k: v for k, v in model_dict_pt.items() if k in key}
            model.load_state_dict(embedding_dict, strict=False) # we can also use nn.Embeddings.from_pretrained()
            if cfg.get("item_freeze"):
                model.net.item_embeddings.requires_grad_(False) # freeze the item
            log.info('Training with only the item embeddings')
            trainer.fit(model=model, datamodule=datamodule)
        else:
            log.info("Training with the whole models")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")) 

    if cfg.get("train") and not cfg.get("oie_learning"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # save the item embeddings as a npz file
    # after training and validation, we will have n-epochs of item embeddings
    # it is stored in model.items which is a list
    # so, we first convert it as numpy array, then save it as npz
    if cfg.get("save_item_embeddings"):
        name = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir.split('/')[-1]
        evol_item_embeddings = np.array(model.items)
        np.savez(cfg.paths.item_embeddings_dir+f'{name}'+'.npz', item_embeddings=evol_item_embeddings)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
