import os
import json
import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import numpy as np
from transformers import BertModel

from shared.myvisual import VisualEncoder

logger = logging.getLogger('root')

class BertForEntity(nn.Module):
    def __init__(self,  args ,num_ner_labels, head_hidden_dim=150, width_embedding_dim=150, max_span_length=8):
        super(BertForEntity,self).__init__()

        self.args=args
        self.vb=VisualEncoder(args)
        self.ner_dropout = nn.Dropout(0.1)
        self.ner_classifier = nn.Sequential(
            nn.Linear(768*2, 300),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(300, num_ner_labels)
        )
    
    #With the same seed, "torch.index_select" can not reproduce the result.
    #So I reload this function. It works!
    def index_select(self,input_tensor,dim,indices):
        tensor_transpose = torch.transpose(input_tensor, 0, dim)
        return tensor_transpose[indices].transpose(dim, 0)

    def forward(self, main_input_ids, main_attention_mask,
                    aux_input_ids,aux_attention_mask,
                    spans, spans_mask, spans_ner_label, 
                    attention_with_image,
                    pixel_values,aux_values, rcnn_values):

        sequence_output = self.vb(main_input_ids=main_input_ids, main_attention_mask=main_attention_mask,attention_with_image=attention_with_image, \
                            aux_input_ids=aux_input_ids,aux_attention_mask=aux_attention_mask, \
                            pixel_values=pixel_values,aux_values=aux_values,rcnn_values=rcnn_values)
        #get start_token hidden
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding=torch.zeros(spans.size(0),spans.size(1),sequence_output.size(-1)).cuda()
        for index,(hiddens,start) in enumerate(zip(sequence_output,spans_start)) :
            span=self.index_select(hiddens,0,start)
            spans_start_embedding[index]=span
        
        #get end_token hidden
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding=torch.zeros(spans.size(0),spans.size(1),sequence_output.size(-1)).cuda()
        for index,(hiddens,end) in enumerate(zip(sequence_output,spans_end)) :
            span=self.index_select(hiddens,0,end)
            spans_end_embedding[index]=span

        spans_embedding = torch.cat((spans_start_embedding, spans_end_embedding), dim=-1)
        
        spans_embedding =self.ner_dropout(spans_embedding)
        logits = self.ner_classifier(spans_embedding)

        if spans_ner_label is not None:
            loss_fct = CrossEntropyLoss(reduction='sum')
            if main_attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss, spans_ner_label.view(-1), torch.tensor(loss_fct.ignore_index).type_as(spans_ner_label)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, logits.shape[-1]), spans_ner_label.view(-1))
            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding
