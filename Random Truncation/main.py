import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
from datetime import datetime
import argparse
import glob
import os
import time
from argparse import Namespace
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Dict, List

from transformers.data.processors.utils import InputFeatures
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities import rank_zero_info
from models.transformers.model import Transformer_PL
import wandb


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
        pl_module.logger.log_metrics(lrs)
        
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Train results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        time=datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, f"{time}_test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))

def add_generic_args(parser, root_dir) -> None:
    #  TODO(SS): allow all pl args? parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--offline", action="store_true", default=False, help="Whether to upload to wandb.")

    parser.add_argument(
        "--max_epochs",
        default=20,
        type=int,
        help="The number of GPUs allocated for this, it is by default 0 meaning none",
    )
    parser.add_argument(
        "--min_epochs",
        default=5,
        type=int,
        help="The number of GPUs allocated for this, it is by default 0 meaning none",
    )

    parser.add_argument(
        "--gpus", #set to 0 if using cpu
        default=1,
        type=int,
        help="The number of GPUs allocated for this, it is by default 0 meaning none",
    )
    parser.add_argument(
            "--max_seq_length",
            default=4096,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--fp16",default=False, #set to False if using cpu
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--do_train", action="store_true", default=False, help="Whether to run training.")
    parser.add_argument("--do_validate", action="store_true", default=False, help="Whether to run validate.")
    parser.add_argument("--do_predict", action="store_true", default=False, help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files.",
    )
    parser.add_argument(
        "--dataset",
        default="yelp",
        type=str,
        help="Pretrained tokenizer name or path",
    )
    
    ## Need to upgrade pytorch lightning to enable lr find
    # parser.add_argument(
    #     "--lr_find",
    #     default=False,
    #     action="store_true",
    #     help="lr finder",
    # )
# define accuracy metrics
# ~ def compute_metrics(pred):
    # ~ labels = pred.label_ids
    # ~ preds = pred.predictions.argmax(-1)
    # ~ # argmax(pred.predictions, axis=1)
    # ~ #pred.predictions.argmax(-1)
    # ~ precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    # ~ acc = accuracy_score(labels, preds)
    # ~ return {
        # ~ 'accuracy': acc,
        # ~ 'f1': f1,
        # ~ 'precision': precision,
        # ~ 'recall': recall
    # ~ }
     
def generic_train(
    model: Transformer_PL,
    args: argparse.Namespace,
    early_stopping_callback=False,
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs
):
    
    # init model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)
    log_dir = Path(os.path.join(model.hparams.output_dir, 'logs'))
    log_dir.mkdir(exist_ok=True)

    # build logger
    # # WanDB logger
    # wandb.init()
    # sweep_id = os.environ.get(wandb.env.SWEEP_ID)
    # run_id = os.environ.get(wandb.env.RUN_ID)
    # experiment = wandb.init()
    # pl_logger = pl_loggers.WandbLogger(
    #     project=f"info-solicitation",
    #     experiment=experiment,
    #     offline=args.offline
    # )


    # Tensorboard logger
    pl_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir,
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
        default_hp_metric=True
    )

    # add custom checkpoints
    ckpt_path = os.path.join(
        args.output_dir, pl_logger.version, "checkpoints",
    )
    # ckpt_path = os.path.join(
    #     "fold_path, "version_xxxx", "checkpoints",
    # )
    #if just test, change pl_logger.version to "version_xxxx" folder that contains the model, also change output dir if needed (from args.output_dir to actual dir)
    if checkpoint_callback is None:
        if args.do_validate == True:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_path, filename="{epoch}-{val_loss:.2f}", monitor="val_loss", mode="min", save_top_k=1, verbose=True
            )
        else:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_path, filename="{epoch}-{train_loss:.2f}", monitor="train_loss", mode="min", save_top_k=-1, verbose=True
            )
    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = {}

    # TODO: remove with PyTorch 1.6 since pl uses native amp
    train_params["max_epochs"] = args.max_epochs
    train_params["min_epochs"] = args.min_epochs
    
    if args.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = args.fp16_opt_level
        # ~ train_params["amp_backend"] = "native"

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        callbacks=[logging_callback] + extra_callbacks,
        logger=pl_logger,
        checkpoint_callback=checkpoint_callback,
        limit_val_batches= 1.0 if args.do_validate == True else 0.0,
        num_sanity_val_steps= 2 if args.do_validate == True else 0,
        val_check_interval=1.0 if args.do_validate == True else args.max_epochs+2,
        check_val_every_n_epoch= 1 if args.do_validate == True else args.max_epochs+2,
        **train_params,
    )

    ## Need to upgrade pytorch lightning to enable lr find
    # if args.lr_find:
    #     lr_finder = trainer.lr_find(model)
    #     fig = lr_finder.plot(suggest=True)
    #     fig.show()
    #     new_lr = lr_finder.suggestion()
    #     logger.info("Recommended Learning Rate: %s", new_lr)

    if args.do_train:
        trainer.fit(model)
        # track model performance under differnt hparams settings in "Hparams" of TensorBoard
        # pl_logger.log_hyperparams(params=model.hparams, metrics={'hp_metric': checkpoint_callback.best_model_score.item()})
        # pl_logger.save()
    return trainer, ckpt_path


def main():
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = Transformer_PL.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    print(args)
    # fix random seed to make sure the result is reproducible
    pl.seed_everything(args.seed)

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)

    model = Transformer_PL(args)
    trainer, ckpt_path = generic_train(model, args)

    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:
        checkpoints = list(sorted(glob.glob(os.path.join(ckpt_path, "epoch=*.ckpt"), recursive=True)))
        print('checkpoints',checkpoints)
        model = model.load_from_checkpoint(checkpoints[-1])
        model.hparams.data_dir = args.data_dir
        model.hparams.max_seq_length = args.max_seq_length
        model.hparams.output_dir = args.output_dir
        return trainer.test(model)


if __name__ == "__main__":
    main()
