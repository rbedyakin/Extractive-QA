# Extractive QA

## Description
This repository hosts the code for Extractive Question Answering.

## Installation
Install all required python dependencies:
```
pip install -r requirements.txt
```

## How to use

### T5 model
```
python main.py --config ./config/t5.yaml
```
### BERT-like models (ModernBERT, DeBERTa, BERT, ...)
```
python main.py --config ./config/modernbert.yaml
```

## Data
HuggingFace dataset page: https://huggingface.co/datasets/xwjzds/extractive_qa_question_answering_hr <br>
Please refer to [HR-MultiWOZ: A Task Oriented Dialogue (TOD) Dataset for HR LLM Agent](https://arxiv.org/abs/2402.01018) for details about the dataset construction.

## Models

- [x] T5 <br>
- [x] ModernBERT <br>
- [ ] LLama 3.2 <br>
- [ ] NuExtract-1.5 <br>
