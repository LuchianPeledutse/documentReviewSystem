import os
import sys
import json
from copy import copy
import uuid
import requests
from typing import List
sys.path.append("..\\")

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import AutoModel, AutoTokenizer

from systemComponents import LlmModelModule
from prompts import suggestions_based_on_technologies

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


EMBEDDING_MODEL_NAME = 'Snowflake/snowflake-arctic-embed-l-v2.0'
TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
EMBD_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)


class NERModelBERT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 embedding_model: AutoTokenizer, num_layers: int = 2, num_tags: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_model = embedding_model
        self.classifier = nn.Linear(input_dim, num_tags)
        self.rnn = BertModel(self.config).encoder

    @property
    def config(self):
        config = BertConfig(
            vocab_size=len(TOKENIZER.get_vocab()),
            hidden_size=self.input_dim,
            intermediate_size = self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=4)
        return config

    def forward(self, x: torch.tensor) -> torch.tensor:
        """x is tensor of shape BxS where B is batch size and S is sequence length of inputs"""
        with torch.no_grad():
            emb_x = self.embedding_model(x).last_hidden_state
        emb_x = torch.permute(emb_x, (1, 0, 2))
        last_hidden_state = self.rnn(emb_x).last_hidden_state
        last_hidden_state = torch.permute(last_hidden_state, (1, 0, 2))
        y = self.classifier(last_hidden_state) # B x S x E
        return y

# Model inference
class ModelInference:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def infer(self, text: str, model: nn.Module, device: str = 'cuda') -> List[str]:
        tokenized_text = TOKENIZER(text).input_ids
        tensor_text = torch.tensor(tokenized_text).unsqueeze(dim = 0).to(device = device)
        prediction = model(tensor_text).squeeze(dim = 0)
        label_prediction = prediction.softmax(dim = 1).argmax(dim = 1).cpu()
        inds_of_2 = (label_prediction == 2).nonzero(as_tuple = True)[0].tolist()
        # Collecting technologies to token_words_list
        token_words_list: List[List[int]] = []
        for single_2ind in inds_of_2:
            current_token_word = []
            if single_2ind == len(label_prediction) - 1:
                break
            current_ind = single_2ind + 1
            current_tag = label_prediction[current_ind]
            current_token_word.append(tokenized_text[current_ind - 1])
            while current_tag == 1:
                word_token = tokenized_text[current_ind]
                current_token_word.append(word_token)
                current_ind += 1
                if current_ind == len(label_prediction):
                    break
                else:
                    current_tag = label_prediction[current_ind]
            if len(current_token_word) not in [0, 1]:
                token_words_list.append(copy(current_token_word))
            current_token_word.clear()
        return self.tokenizer.decode(token_words_list, skip_special_tokens = True)
    

class Text(BaseModel):
    text: str

class Entities(BaseModel):
   entities: List[str]


ner_model = NERModelBERT(input_dim = 1024, hidden_dim = 256, 
                         num_layers = 2, embedding_model = EMBD_MODEL).to(device='cuda')
ner_model.rnn.load_state_dict(torch.load("C:\\main\\GitHub\\documentReviewSystem\\NERTraining\\models\\BERT_MODEL_epoch#61.pth"))
model_inference = ModelInference(TOKENIZER)

app = FastAPI()

@app.post("/entities")
def get_entities(text_model: Text) -> Entities:
    entities = model_inference.infer(text_model.text, ner_model, device = 'cuda')
    return {"entities": entities if type(entities) == list else []}

if __name__ == "__main__":
    uvicorn.run(app, port = 8002)


