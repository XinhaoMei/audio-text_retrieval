#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import math
import torch
import torch.nn as nn
import numpy as np
from models.BERT_Config import MODELS



class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()

        bert_type = config.bert_encoder.type
        dropout = config.training.dropout

        self.tokenizer = MODELS[bert_type][1].from_pretrained(bert_type)
        if 'clip' not in bert_type:
            self.bert_encoder = MODELS[bert_type][0].from_pretrained(bert_type,
                                                                     add_pooling_layer=False,
                                                                     hidden_dropout_prob=dropout,
                                                                     attention_probs_dropout_prob=dropout,
                                                                     output_hidden_states=False)
        else:
            self.bert_encoder = MODELS[bert_type][0].from_pretrained(bert_type)

        if config.training.freeze:
            for name, param in self.bert_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, captions):
        # device = next(self.parameters()).device
        device = torch.device('cuda')
        tokenized = self.tokenizer(captions, add_special_tokens=True,
                                   padding=True, return_tensors='pt')
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        output = self.bert_encoder(input_ids=input_ids,
                                   attention_mask=attention_mask)[0]

        cls = output[:, 0, :]
        return cls

