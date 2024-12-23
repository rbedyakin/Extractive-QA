import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import lightning as L
from typing import Tuple, List
from torch.optim.adamw import AdamW
from transformers.tokenization_utils_base import BatchEncoding
from src.qa_metrics import QA_Metric
from src.f1_loss import f1_loss


class T5QAModel(L.LightningModule):

    def __init__(
        self,
        pretrained_model_name: str,
        batch_size: int,
        learning_rate: float,
    ):
        super().__init__()
        # self.save_hyperparameters()

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name,
            return_dict=True)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)

        # self.train_metrics = QA_Metric(stage="train")
        self.valid_metrics = QA_Metric(stage="valid")
        self.test_metrics = QA_Metric(stage="test")
        # self.criterion = F.cross_entropy
        # self.criterion = f1_loss

    def forward(self,
                input_ids,
                attention_mask,
                decoder_attention_mask=None,
                labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch: BatchEncoding,
                      batch_idx: int) -> torch.Tensor:

        train_loss, outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )

        # train_loss = self.criterion(logits, batch["labels"])
        self.log("train_loss", train_loss, prog_bar=True)

        return train_loss

    def on_train_epoch_end(self):
        # self.train_metrics.reset()
        pass

    def validation_step(self, batch: BatchEncoding, batch_idx: int):
        val_loss, outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )

        # val_loss = self.criterion(logits, batch["labels"])
        self.log("val_loss", val_loss, prog_bar=True)

        predictions = self.predict_step(batch=batch, batch_idx=batch_idx)
        lm_labels = batch['labels']
        lm_labels[lm_labels[:, :] == -100] = self.tokenizer.pad_token_id
        target = self.tokenizer.batch_decode(lm_labels,
                                             skip_special_tokens=True)

        self.valid_metrics.update(predictions, target)

    def on_validation_epoch_end(self):
        output = self.valid_metrics.compute()
        self.log_dict(output)
        # self.log('valid_chrf_score', output)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        test_loss, outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )

        # test_loss = self.criterion(logits, batch["labels"])
        self.log("test_loss", test_loss, prog_bar=True)

        predictions = self.predict_step(batch=batch, batch_idx=batch_idx)
        lm_labels = batch['labels']
        lm_labels[lm_labels[:, :] == -100] = self.tokenizer.pad_token_id
        target = self.tokenizer.batch_decode(lm_labels,
                                             skip_special_tokens=True)

        self.test_metrics.update(predictions, target)

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.log_dict(output)
        # self.log('test_chrf_score', output)
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx):

        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            num_beams=1,
            max_length=32,
            repetition_penalty=2.5,
            length_penalty=None,
            # early_stopping=True,
            # top_p=0.95,
            # top_k=50,
            num_return_sequences=1,
        )
        # if num_return_sequences>1, then batch_decode
        # returns batch_size * num_return_sequences results
        predictions = self.tokenizer.batch_decode(generated_ids,
                                                  skip_special_tokens=True)
        return predictions

    def configure_optimizers(self) -> AdamW:
        """ configure optimizers """
        # return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.01,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        return [optimizer], [
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        ]


class BertQAModel(L.LightningModule):

    def __init__(
        self,
        pretrained_model_name: str,
        batch_size: int,
        learning_rate: float,
    ):
        super().__init__()
        # self.save_hyperparameters()

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.pretrained_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name)
        self.pretrained_model.train()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        self.classifier = nn.Linear(self.pretrained_model.config.hidden_size,
                                    2)

        self.train_metrics = QA_Metric(stage="train")
        self.valid_metrics = QA_Metric(stage="valid")
        self.test_metrics = QA_Metric(stage="test")

        self.criterion = f1_loss

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """ forward step """
        outputs = self.pretrained_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        output_hidden_states=True)
        h_cls = outputs.hidden_states[-1]
        logits = self.classifier(h_cls)

        return logits

    def prepare_logits_for_metrics(
            self, logits: torch.Tensor, labels: torch.Tensor,
            input_ids) -> Tuple[List[List[str]], List[List[str]]]:

        predictions = torch.argmax(logits, dim=-1)

        true_ids = [[ids for l, ids in zip(label, input_id) if l == 1]
                    for label, input_id in zip(labels, input_ids)]
        true_labels = self.tokenizer.batch_decode(
            sequences=true_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)

        pred_ids = [[ids for p, ids in zip(prediction, input_id) if p == 1]
                    for prediction, input_id in zip(predictions, input_ids)]
        pred_labels = self.tokenizer.batch_decode(
            sequences=pred_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)

        return pred_labels, true_labels

    def training_step(self, batch: BatchEncoding,
                      batch_idx: int) -> torch.Tensor:

        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        train_loss = self.criterion(logits, batch["labels"])
        self.log("train_loss", train_loss, prog_bar=True)

        preds, target = self.prepare_logits_for_metrics(
            logits=logits,
            labels=batch["labels"],
            input_ids=batch["input_ids"])

        batch_value = self.train_metrics(preds, target)
        self.log_dict(batch_value)

        return train_loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()
        # pass

    def validation_step(self, batch: BatchEncoding, batch_idx: int):
        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        val_loss = self.criterion(logits, batch["labels"])
        self.log("val_loss", val_loss, prog_bar=True)

        preds, target = self.prepare_logits_for_metrics(
            logits=logits,
            labels=batch["labels"],
            input_ids=batch["input_ids"])

        self.valid_metrics.update(preds, target)

    def on_validation_epoch_end(self):
        output = self.valid_metrics.compute()
        self.log_dict(output)
        # self.log('valid_chrf_score', output)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        test_loss = self.criterion(logits, batch["labels"])
        self.log("test_loss", test_loss, prog_bar=True)

        preds, target = self.prepare_logits_for_metrics(
            logits=logits,
            labels=batch["labels"],
            input_ids=batch["input_ids"])

        self.test_metrics.update(preds, target)

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.log_dict(output)
        # self.log('test_chrf_score', output)
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx):

        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        predictions = torch.argmax(logits, dim=-1)

        pred_ids = [[
            ids for p, ids in zip(prediction, input_id) if p == 1
        ] for prediction, input_id in zip(predictions, batch['input_ids'])]

        predictions = self.tokenizer.batch_decode(
            sequences=pred_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)

        return predictions

    def configure_optimizers(self) -> AdamW:
        """ configure optimizers """
        # return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
