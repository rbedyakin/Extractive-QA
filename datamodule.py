from transformers import AutoTokenizer, AutoModel, T5Tokenizer
import lightning as L
from datasets import load_dataset
import torch
from transformers import DataCollatorForTokenClassification, DataCollatorForSeq2Seq
from datasets.formatting.formatting import LazyBatch
from torch.utils.data.dataloader import DataLoader
from transformers.tokenization_utils_base import BatchEncoding
from typing import List, Optional, Tuple
import re


class TokenClassificationData(L.LightningDataModule):

    def __init__(self, dataset_path: str, pretrained_model_name: str,
                 batch_size: int, num_workers: int):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_path = dataset_path
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def tokenize_data(self, examples):
        tokens = []
        labels = []
        for question, context, answer in zip(examples['question'],
                                             examples['answer_context'],
                                             examples['answer']):
            input = f"Context: {context} Question: {question} </s>"
            output = f"{answer} </s>"
            tokens.append(input.split())
            labels.append(output.split())

        source_text_encoding = self.tokenizer(
            tokens,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            is_split_into_words=True,
            return_tensors="pt",
        )

        target_text_encoding = self.tokenizer(
            labels,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            is_split_into_words=True,
            return_tensors="pt",
        )

        lm_labels = target_text_encoding['input_ids']
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_text_encoding['input_ids'],
            'attention_mask': source_text_encoding['attention_mask'],
            'decoder_attention_mask': target_text_encoding['attention_mask'],
            'labels': target_text_encoding['input_ids'],
        }

    def prepare_data(self):
        # load_dataset(path="json", data_files=self.data_files)
        pass

    def setup(self, stage: str):
        if stage == "fit":
            raw_datasets = load_dataset(path=self.dataset_path)

            tokenized_datasets = raw_datasets.map(
                self.tokenize_data,
                batched=True,
                remove_columns=raw_datasets['train'].column_names,
            )

            datasets_split = tokenized_datasets['train'].train_test_split(
                test_size=0.1)
            self.test_dataset = datasets_split["test"]
            datasets_split_again = datasets_split["train"].train_test_split(
                test_size=0.1)

            self.train_dataset = datasets_split_again["train"]
            self.val_dataset = datasets_split_again["test"]

            self.data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding="max_length",
                label_pad_token_id=-100,
                max_length=512)

    def train_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           collate_fn=self.data_collator,
                                           num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           collate_fn=self.data_collator,
                                           num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           collate_fn=self.data_collator,
                                           num_workers=self.num_workers)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           collate_fn=self.data_collator,
                                           num_workers=self.num_workers)
