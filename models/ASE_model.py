#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tools.utils import l2norm
from models.AudioEncoder import Cnn10, ResNet38, Cnn14
from models.TextEncoder import BertEncoder, W2VEncoder
from models.BERT_Config import MODELS


class AudioEnc(nn.Module):

    def __init__(self, config):
        super(AudioEnc, self).__init__()

        if config.cnn_encoder.model == 'Cnn10':
            self.audio_enc = Cnn10(config)
        elif config.cnn_encoder.model == 'ResNet38':
            self.audio_enc = ResNet38(config)
        elif config.cnn_encoder.model == 'Cnn14':
            self.audio_enc = Cnn14(config)
        else:
            raise NotImplementedError('No such audio encoder network.')

        if config.cnn_encoder.pretrained:
            # loading pretrained CNN weights
            pretrained_cnn = torch.load('pretrained_models/audio_encoder/{}.pth'.
                                        format(config.cnn_encoder.model))['model']
            dict_new = self.audio_enc.state_dict().copy()
            trained_list = [i for i in pretrained_cnn.keys()
                            if not ('fc' in i or i.startswith('spec') or i.startswith('logmel'))]
            for i in range(len(trained_list)):
                dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
            self.audio_enc.load_state_dict(dict_new)
        if config.training.freeze:
            for name, param in self.audio_enc.named_parameters():
                param.requires_grad = False

    def forward(self, inputs):
        audio_encoded = self.audio_enc(inputs)
        return audio_encoded


class ASE(nn.Module):

    def __init__(self, config):
        super(ASE, self).__init__()

        self.l2 = config.training.l2
        joint_embed = config.joint_embed

        self.audio_enc = AudioEnc(config)

        if config.cnn_encoder.model == 'Cnn10':
            self.audio_linear = nn.Sequential(
                nn.Linear(512, joint_embed),
                nn.ReLU(),
                nn.Linear(joint_embed, joint_embed)
            )
        elif config.cnn_encoder.model == 'ResNet38' or config.cnn_encoder.model == 'Cnn14':
            self.audio_linear = nn.Sequential(
                nn.Linear(2048, joint_embed * 2),
                nn.ReLU(),
                nn.Linear(joint_embed * 2, joint_embed)
            )

        # self.audio_gated_linear = nn.Linear(joint_embed, joint_embed)
        if config.text_encoder == 'bert':
            self.text_enc = BertEncoder(config)
            bert_type = config.bert_encoder.type
            self.text_linear = nn.Sequential(
                nn.Linear(MODELS[bert_type][2], joint_embed * 2),
                nn.ReLU(),
                nn.Linear(joint_embed * 2, joint_embed)
            )
        elif config.text_encoder == 'w2v':
            self.text_enc = W2VEncoder(config)
            self.text_linear = nn.Sequential(
                nn.Linear(300, joint_embed),
                nn.ReLU(),
                nn.Linear(joint_embed, joint_embed)
            )

    def encode_audio(self, audios):
        return self.audio_enc(audios)

    def encode_text(self, captions):
        return self.text_enc(captions)

    def forward(self, audios, captions):

        audio_encoded = self.encode_audio(audios)     # batch x channel
        caption_encoded = self.encode_text(captions)

        audio_embed = self.audio_linear(audio_encoded)

        caption_embed = self.text_linear(caption_encoded)

        if self.l2:
            # apply l2-norm on the embeddings
            audio_embed = l2norm(audio_embed)
            caption_embed = l2norm(caption_embed)

        return audio_embed, caption_embed
