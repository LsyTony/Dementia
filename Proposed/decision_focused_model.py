
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
import argparse
import os
import csv
import random
import math
import torch
from pathlib import Path
from typing import Any, Dict
from sklearn.metrics import classification_report
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities import rank_zero_info
from models.utils import load_jsonl_gz
from decision_focused_dataset import DecisionFocusedDataset
from decision_focused_selection import DecisionFocusedSentenceSelector
import torchmetrics

from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}


# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

class DecisionFocusedTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels=None,
        config=None,
        tokenizer=None,
        model=None,
        meta_iterations=10,
        **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        # TODO: move to self.save_hyperparameters()
        # self.save_hyperparameters()
        # can also expand arguments into trainer signature for easier reading
        mode="sequence-classification"
        
        self.save_hyperparameters(hparams)
        logger.info(f"Number of Labels: {self.hparams.num_labels}")

        self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        self.meta_iterations = meta_iterations
        self.current_iteration = 1

        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                **({"num_labels": self.hparams.num_labels} if self.hparams.num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
            print(self.config)
        else:
            self.config: PretrainedConfig = config

        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(self.config, p), f"model config doesn't have a `{p}` attribute"
                setattr(self.config, p, getattr(self.hparams, p))

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer

        self.model_type = MODEL_MODES[mode]
        if model is None:
            self.model = self.model_type.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=self.config,
                cache_dir=cache_dir, # save server storage
            )
        else:
            self.model = model
        #add metrics
        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1(num_classes=self.hparams.num_labels,average="macro")
        self.train_precision = torchmetrics.Precision(num_classes=self.hparams.num_labels,average="macro")
        self.train_recall = torchmetrics.Recall(num_classes=self.hparams.num_labels,average="macro")
        self.val_acc = torchmetrics.Accuracy()
        self.val_f1 = torchmetrics.F1(num_classes=self.hparams.num_labels,average="macro")
        self.val_precision = torchmetrics.Precision(num_classes=self.hparams.num_labels,average="macro")
        self.val_recall = torchmetrics.Recall(num_classes=self.hparams.num_labels,average="macro")
        self.test_acc = torchmetrics.Accuracy()
        self.test_f1 = torchmetrics.F1(num_classes=self.hparams.num_labels,average="macro")
        self.test_precision = torchmetrics.Precision(num_classes=self.hparams.num_labels,average="macro")
        self.test_recall = torchmetrics.Recall(num_classes=self.hparams.num_labels,average="macro")
        
        # Initialize decision-focused sentence selector
        self.sentence_selector = DecisionFocusedSentenceSelector(
            max_tokens=self.hparams.max_seq_length,
            relevance_threshold=1.0,  # Initial threshold for first iteration
            iteration=self.current_iteration,
            max_iterations=self.meta_iterations
        )
        
        # Dictionary to track sentence relevance scores
        self.sentence_index_tracker = {}
        
        
    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer

        self.train_loader = getattr(self,"train_loader",None)
        if self.train_loader:
            scheduler = self.get_lr_scheduler()
        else:
            return [optimizer]
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # Extract batch components
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        patient_ids = batch[5]  # List of patient IDs

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        loss= outputs[0]
        logits = outputs[1]# outputformate is loss, logits contains vectors before softmax, hidden state, attentions
        y_pred = logits.argmax(-1) #no need for detach cpu numpy becasue torchmetrics works on tensor directly
        y_tgt = inputs["labels"] #format is tensor
        
        # Update sentence relevance scores based on prediction correctness
        for i in range(len(y_pred)):
            patient_id = patient_ids[i]
            prediction_correct = y_pred[i] == y_tgt[i]
            
            # Update relevance scores in the sentence selector
            self.sentence_selector.update_relevance_scores(patient_id, prediction_correct)
            
            # Also update the tracking dictionary for compatibility with existing code
            dict_key = str(patient_id)
            if dict_key in self.sentence_index_tracker:
                self.sentence_index_tracker[dict_key] += 1
            else:
                self.sentence_index_tracker[dict_key] = 1
                
            if prediction_correct:
                dict_key = str(patient_id) + '_true'
                if dict_key in self.sentence_index_tracker:
                    self.sentence_index_tracker[dict_key] += 1
                else:
                    self.sentence_index_tracker[dict_key] = 1

        #accumulate and return metrics for logging
        acc = self.train_acc(y_pred,y_tgt)
        f1 = self.train_f1(y_pred,y_tgt)
        #just accumulate
        self.train_precision.update(y_pred,y_tgt)
        self.train_recall.update(y_pred,y_tgt)
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        self.log('train_loss', loss, prog_bar=True)
        self.log( "rate", lr_scheduler.get_last_lr()[-1])
        self.log('train_accuracy', acc, prog_bar=True)
        self.log('train_f1',f1)
        self.log('current_iteration', self.current_iteration)
        self.log('relevance_threshold', self.sentence_selector.relevance_threshold)
        #tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        #print('sentence score tracker',self.sentence_index_tracker)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        #print('whole batch info before getting inputs',batch)
        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None
        #print('inputs to model',inputs)
        outputs = self(**inputs)
        #print('output from model',outputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.argmax(-1)
        out_label_ids = inputs["labels"]

        eval_acc = self.val_acc(preds,out_label_ids)
        self.val_f1.update(preds,out_label_ids)
        self.val_precision.update(preds,out_label_ids)
        self.val_recall.update(preds,out_label_ids)
        
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
        self.log('val_loss', tmp_eval_loss, prog_bar=True)
        self.log('val_accuracy', eval_acc, prog_bar=True)
        #print("val_loss",tmp_eval_loss.detach().cpu(), "pred", preds, "target", out_label_ids)
        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_nb):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.argmax(-1)
        out_label_ids = inputs["labels"]
        self.test_acc.update(preds,out_label_ids)
        self.test_f1.update(preds,out_label_ids)
        self.test_precision.update(preds,out_label_ids)
        self.test_recall.update(preds,out_label_ids)
        #preds = logits.detach().cpu().numpy()
        #out_label_ids = inputs["labels"].detach().cpu().numpy()
        self.log('test_loss', tmp_eval_loss)
        return preds, out_label_ids
        
    def train_epoch_end(self, outputs):
        print('sentence score tracker',self.sentence_index_tracker)

        # Export selected sentences to CSV
        csv_path = os.path.join(
            self.output_dir, 
            f"selected_sentences_iteration_{self.current_iteration}_epoch_{self.current_epoch}.csv"
        )
        self.sentence_selector.export_selected_sentences_to_csv(csv_path)
        logger.info(f"Exported selected sentences to {csv_path}")

        # Adjust threshold for next epoch using simulated annealing
        # Create a small subset of the training data for threshold evaluation
        subset_loader = self.get_subset_dataloader()
        if subset_loader is not None:
            new_threshold = self.sentence_selector.adjust_threshold(self.model, subset_loader)
            logger.info(f"Adjusted relevance threshold from {self.sentence_selector.relevance_threshold} to {new_threshold}")
        
        #compute metrics
        train_accuracy = self.train_acc.compute()
        train_f1 = self.train_f1.compute()
        train_precision = self.train_precision.compute()
        train_recall = self.train_recall.compute()
        #log metrics
        self.log("epoch_train_accuracy", train_accuracy)
        self.log("epoch_train_f1", train_f1)
        self.log("epoch_train_precision", train_precision)
        self.log("epoch_train_recall", train_recall)
        #reset all metrics
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        print(f"\ntraining accuracy: {train_accuracy:.4}, "\
        f"f1: {train_f1:.4}, precision: {train_precision:.4}, recall:{train_recall:.4}")

        # If we've reached the max epochs for this iteration of the meta-algorithm,
        # move to the next iteration
        if (self.current_epoch + 1) % (self.hparams.max_epochs // self.meta_iterations) == 0:
            self.current_iteration += 1
            if self.current_iteration <= self.meta_iterations:
                self.sentence_selector.next_iteration()
                logger.info(f"Moving to meta-algorithm iteration {self.current_iteration}")
                
    def validation_epoch_end(self, outputs):
		#compute metrics
        val_accuracy = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        #log metrics
        self.log("val_accuracy", val_accuracy)
        self.log("val_f1", val_f1)
        self.log("val_precision", val_precision)
        self.log("val_recall", val_recall)
        #reset all metrics
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        print(f"\nvalidation accuracy: {val_accuracy:.4}, "\
        f"f1: {val_f1:.4}, precision: {val_precision:.4}, recall:{val_recall:.4}")
        
    def test_epoch_end(self, outputs):
        #compute metrics
        test_accuracy = self.test_acc.compute()
        test_f1 = self.test_f1.compute()
        test_precision = self.test_precision.compute()
        test_recall = self.test_recall.compute()
        #log metrics
        self.log("test_accuracy", test_accuracy)
        self.log("test_f1", test_f1)
        self.log("test_precision", test_precision)
        self.log("test_recall", test_recall)
        #reset all metrics
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        #classification_report
        preds = None
        tgts = None
        for pred, tgt in outputs:
            preds = torch.cat([preds, pred]) if preds is not None else pred
            tgts = torch.cat([tgts, tgt]) if tgts is not None else tgt
        final_preds = preds.detach().cpu().numpy()
        final_tgts = tgts.detach().cpu().numpy()
        class_names = ['case','control']
        print(f"\ntest accuracy: {test_accuracy:.4}, "\
        f"f1: {test_f1:.4}, precision: {test_precision:.4}, recall:{test_recall:.4}")
        print(classification_report(final_tgts, final_preds, target_names=class_names))

    def get_subset_dataloader(self, sample_size=100):
        if not hasattr(self, 'train_dataloader') or self.train_dataloader() is None:
            return None
            
        # Get original dataset
        dataset = self.train_dataloader().dataset
        
        # Check if we have enough samples
        if len(dataset) <= sample_size:
            return self.train_dataloader()
            
        # Sample random indices
        indices = random.sample(range(len(dataset)), sample_size)
        subset_data = [dataset.data[i] for i in indices]
        
        # Create new dataset with subset
        subset_dataset = DecisionFocusedDataset(
            tokenizer=self.tokenizer,
            data=subset_data,
            max_len=self.hparams.max_seq_length,
            selector=self.sentence_selector
        )
        
        # Create and return dataloader
        return torch.utils.data.DataLoader(
            dataset=subset_dataset,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=subset_dataset.collate_fn
        )
    
    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        # dataset_size = len(self.train_loader.dataset)
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    def setup(self, mode):
        if mode == "fit":
            self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)
            self.dataset_size = len(self.train_dataloader().dataset)
        elif mode == "test":
            self.dataset_size = len(self.test_dataloader().dataset)
			#test=next(iter(self.train_loader))
            #print('train_loader example',test)
            # ~ print(self.train_loader['review'])
            # ~ print(self.train_loader['input_ids'].shape)
            # ~ print(self.train_loader['attention_mask'].shape)
            # ~ print(self.train_loader['targets'].shape)
            
    def get_dataloader(self, type_path, batch_size, shuffle=False):
        # todo add dataset path
        data_filepath = os.path.join(self.hparams.data_dir, type_path+".jsonl.gz")
        data = load_jsonl_gz(data_filepath)
        
        data_set = DecisionFocusedDataset(
            self.tokenizer,
            data,
            self.hparams.max_seq_length,
            selector=self.sentence_selector
        )
        logger.info(f"Loading {type_path} dataset with length {len(data_set)} from {data_filepath}")
        data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=self.hparams.num_workers,
                                                collate_fn=data_set.collate_fn)
        
        return data_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)


    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--num_labels",
            default=2,
            type=int,
            help="Pretrained tokenizer name or path",
        )
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=10e-4, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_workers", default=16, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument("--train_batch_size", default=4, type=int)
        parser.add_argument("--eval_batch_size", default=4, type=int)
        parser.add_argument("--adafactor", action="store_true")

        return parser
