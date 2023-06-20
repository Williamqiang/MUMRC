import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel

import torch.nn.functional as F

from shared.myvisual import VisualEncoder

class BertForRelation(nn.Module):
    def __init__(self,args, num_rel_labels):
        super(BertForRelation, self).__init__()
        self.num_labels = num_rel_labels
        self.vb=VisualEncoder(args)
        self.re_dropout = nn.Dropout(0.1)
        self.re_classifier = nn.Linear(768 * 2, self.num_labels)

    def forward(self,
                main_input_ids, main_attention_mask,aux_input_ids,aux_attention_mask,attention_with_image,
                labels=None, 
                sub_idx=None, 
                obj_idx=None, 
                pixel_values=None,
                aux_values=None,
                rcnn_values=None,
                input_position=None,
                mode="train"):

        sequence_output = self.vb(main_input_ids=main_input_ids, main_attention_mask=main_attention_mask,
                                 aux_input_ids=aux_input_ids,aux_attention_mask=aux_attention_mask,attention_with_image=attention_with_image,
                                pixel_values=pixel_values,aux_values=aux_values,rcnn_values=rcnn_values)

        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.re_dropout(rep)
        logits = self.re_classifier(rep)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
