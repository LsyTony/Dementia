import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
from datetime import datetime
import argparse
import os
import pandas as pd
from pathlib import Path
from typing import Any, Dict
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from models.transformers.dataloader import TransformerYelpDataset
from models.utils import load_jsonl_gz
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np

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
import torchmetrics

class Transformer_PL(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels=None,
        config=None,
        tokenizer=None,
        model=None,
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
        #self.tokenizer.truncation_side = "left"
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
        #add cls_token
        self.cls_token_list=None
        #add input text
        self.input_text=None

        
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
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        loss= outputs[0]
        logits = outputs[1]# outputformate is loss, logits contains vectors before softmax, hidden state, attentions
        y_pred = logits.argmax(-1) #no need for detach cpu numpy becasue torchmetrics works on tensor directly
        y_tgt = inputs["labels"] #format is tensor
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
        #tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
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
        #print('logits',logits)
        #probs = torch.nn.functional.softmax(logits, dim=1)
        #print('probs', probs)
        #print('probs positive',probs[:,0])
        preds = logits.argmax(-1)
        #print('preds',preds)
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

        outputs = self(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        CLS_tokens = last_hidden_states[:,0,:]
        #print('cls',CLS_tokens,'cls size',CLS_tokens.size())
        #print('cls numpy',CLS_tokens.detach().cpu().numpy(),'cls numpy ize',CLS_tokens.detach().cpu().numpy().shape)
        self.cls_token_list= np.concatenate((self.cls_token_list,CLS_tokens.detach().cpu().numpy())) if \
        self.cls_token_list is not None else CLS_tokens.detach().cpu().numpy()
        #print('token_list_shape',self.cls_token_list.shape)
        #print('batch',batch[5])
        input_text_batch=batch[5]
        self.input_text=np.concatenate((self.input_text,input_text_batch)) if \
        self.input_text is not None else input_text_batch
        tmp_eval_loss, logits = outputs[:2]
        #print('output from model',outputs)
        preds = logits.argmax(-1)
        out_label_ids = inputs["labels"]
        out_study_ids = list(batch[4])
        #print('study_ids',out_study_ids)
        probs = torch.nn.functional.softmax(logits, dim=1)
        self.test_acc.update(preds,out_label_ids)
        self.test_f1.update(preds,out_label_ids)
        self.test_precision.update(preds,out_label_ids)
        self.test_recall.update(preds,out_label_ids)
        #preds = logits.detach().cpu().numpy()
        #out_label_ids = inputs["labels"].detach().cpu().numpy()
        self.log('test_loss', tmp_eval_loss)
        return preds, out_label_ids, probs, out_study_ids, logits
        
    def train_epoch_end(self, outputs):
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
        probs = None
        study_ids = None
        logits = None
        for pred, tgt, prob, study_id, logit in outputs:
            preds = torch.cat([preds, pred]) if preds is not None else pred
            tgts = torch.cat([tgts, tgt]) if tgts is not None else tgt
            probs = torch.cat([probs, prob]) if probs is not None else prob
            study_ids = study_ids+study_id if study_ids is not None else study_id
            logits = torch.cat([logits, logit]) if logits is not None else logit
        final_preds = preds.detach().cpu().numpy()
        final_tgts = tgts.detach().cpu().numpy()
        final_probs = probs.detach().cpu().numpy()
        final_study_ids = study_ids
        final_logits= logits.detach().cpu().numpy()   
        #print('final study_ids', final_study_ids)
        #print('cls tokens size', self.cls_token_list.shape)
        embedding_longformer=pd.DataFrame(self.cls_token_list)
        embedding_longformer.insert(0, "ID", final_study_ids)
        arr_logits=pd.DataFrame(final_logits)
        arr_logits.columns=['predict_logits_1','predict_logits_2']
        arr_probs=pd.DataFrame(final_probs)
        arr_probs.columns=['predict_probs_1','predict_probs_2']
        embedding_longformer= pd.concat([embedding_longformer,arr_logits],axis=1)
        embedding_longformer= pd.concat([embedding_longformer,arr_probs],axis=1)
        embedding_longformer.insert(embedding_longformer.shape[1], "predict", final_preds)
        embedding_longformer.insert(embedding_longformer.shape[1], "label", final_tgts)
        embedding_longformer.to_csv(self.hparams.output_dir+"/embedding_longformer.csv")

        embedding_longformer_text=pd.DataFrame(self.input_text)
        embedding_longformer_text.insert(0, "ID", final_study_ids)
        embedding_longformer_text.insert(embedding_longformer_text.shape[1], "predict", final_preds)
        embedding_longformer_text.insert(embedding_longformer_text.shape[1], "label", final_tgts)
        embedding_longformer_text.to_csv(self.hparams.output_dir+"/embedding_longformer_text.csv")
        print(embedding_longformer)
        class_names = ['case','control'] 
        print(f"\ntest accuracy: {test_accuracy:.4}, "\
        f"f1: {test_f1:.4}, precision: {test_precision:.4}, recall:{test_recall:.4}")
        print(classification_report(final_tgts, final_preds, target_names=class_names, digits=4))
        #self.log("classification report", classification_report(final_tgts, final_preds, target_names=class_names, digits=4))
        conf_mat = confusion_matrix(final_tgts, final_preds)
        n_classes = conf_mat.shape[0]
        for i in range(n_classes):
            tp = conf_mat[i, i]
            fn = sum(conf_mat[i, :]) - tp
            fp = sum(conf_mat[:, i]) - tp
            tn = sum(sum(conf_mat)) - tp - fn - fp
            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)
            print(f"Class {i}: sensitivity = {tpr:.4f}, specificity = {tnr:.4f}")
            self.log_dict({f"Class {i}:sensitivity": tpr, f"Class {i}:specificity":tnr})
        TGT = pd.DataFrame(final_tgts)
        PROBS = pd.DataFrame(final_probs)
        TGT.to_csv(self.hparams.output_dir+"/true_label.csv")
        PROBS.to_csv(self.hparams.output_dir+"/probs.csv")
        # ~ skplt.metrics.plot_roc(final_tgts, final_probs)
        # ~ plt.savefig("roc"+datetime.now().strftime("%d-%m-%Y--%H-%M-%S")+".png")
        # ~ auc_score_0 = roc_auc_score(final_tgts, final_probs[:,0])
        # ~ auc_score_1 = roc_auc_score(final_tgts, final_probs[:,1])
        # ~ print(f"ROC AUC positive: {auc_score_0:.4f}"+f"ROC AUC negative: {auc_score_1:.4f}")
        # ~ skplt.metrics.plot_confusion_matrix(final_tgts, final_preds, normalize=True)
        # ~ plt.savefig("confusion_matrix"+datetime.now().strftime("%d-%m-%Y--%H-%M-%S")+".png")

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        #dataset_size = len(self.train_loader.dataset)
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
        yelp = TransformerYelpDataset(self.tokenizer,data,self.hparams.max_seq_length)
        logger.info(f"Loading {type_path} dataset with length {len(yelp)} from {data_filepath}")
        data_loader = torch.utils.data.DataLoader(dataset=yelp,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=self.hparams.num_workers,
                                                collate_fn=yelp.collate_fn)
        
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
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-3, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_workers", default=16, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument("--train_batch_size", default=4, type=int)
        parser.add_argument("--eval_batch_size", default=4, type=int)
        parser.add_argument("--adafactor", action="store_true")

        return parser
