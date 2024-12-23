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


class T5QAData(L.LightningDataModule):

    def __init__(self, dataset_path: str, pretrained_model_name: str,
                 batch_size: int, num_workers: int):
        super().__init__()
        # self.save_hyperparameters()

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


class BertQAData(L.LightningDataModule):

    def __init__(self, dataset_path: str, pretrained_model_name: str,
                 batch_size: int, num_workers: int):
        super().__init__()
        # self.save_hyperparameters()

        self.dataset_path = dataset_path
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def align_labels_with_tokens(self, labels: List[int],
                                 word_ids: List[Optional[int]]) -> List[int]:
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = 0 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(0)
            else:
                # Same word as previous token
                label = labels[word_id]
                new_labels.append(label)

        return new_labels

    def tokenize_data(self, examples):
        tokens = []
        labels = []
        for question, context, answer in zip(examples['question'],
                                             examples['answer_context'],
                                             examples['answer']):
            input = f"Context: {context} Question: {question}"
            output = f"{answer}"
            input_list = input.split()
            output_list = output.split()
            input = ' '.join(input_list)
            output = ' '.join(output_list)
            pos = input.index(output)
            first_occurrence = 0
            s = 0
            while s < pos:
                s += len(input_list[first_occurrence])
                s += 1  # space
                first_occurrence += 1

            tokens.append(input_list)
            labels_output = [0 for _, _ in enumerate(input_list)]
            for delta, _ in enumerate(output_list):
                labels_output[first_occurrence + delta] = 1
            labels.append(labels_output)

        tokenized_inputs = self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            is_split_into_words=True,
            return_tensors="pt",
            # max_length=self.tokenizer.model_max_length,
            max_length=1024,
        )

        new_labels = []
        for i, labels_i in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(
                labels_i, word_ids))

        tokenized_inputs["labels"] = new_labels

        return tokenized_inputs

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

            self.data_collator = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer,
                padding="max_length",
                max_length=1024,
                label_pad_token_id=0,
            )

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


if __name__ == "__main__":
    data = BertQAData(
        dataset_path="xwjzds/extractive_qa_question_answering_hr",
        pretrained_model_name='google-bert/bert-base-cased',
        batch_size=4,
        num_workers=2)

    data.setup(stage='fit')

    # import torch.nn.functional as F
    # for item in data.train_dataloader():
    #     v = torch.argmax(item['labels'], dim=-1)
    #     print(v)
    #     print(item['labels'][:,v])
    #     y = F.one_hot(item['labels'], num_classes=2)
    #     print(y)
    #     input()
