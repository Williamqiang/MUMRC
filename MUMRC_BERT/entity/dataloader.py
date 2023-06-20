import json
import os 
import sys
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from PIL import Image
import numpy as np
import math
import random
from shared.loadimg import loading
from shared.question_template import visual_entity_question as que 
visual_entity_question=que

class MNREDateset(Dataset):
    def __init__(self,samples,tokenizer,mode,args=None):
        self.args=args
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_context_len=args.max_context_length
        self.mode = mode

    def __getitem__(self,index):
        sample=self.samples[index]
        tokens = sample['tokens']
        spans = sample['spans']
        spans_ner_label = sample['spans_label']
        main_tokens_tensor,aux_tokens_tensor, main_attention_mask,aux_attention_mask,bert_spans_tensor, spans_ner_label_tensor ,attention_with_image= self._get_input_tensors(tokens, spans, spans_ner_label)

        assert(main_tokens_tensor.shape[0] == main_attention_mask.shape[0]) 
        assert(bert_spans_tensor.shape[0] == spans_ner_label_tensor.shape[0]) 
        assert(aux_tokens_tensor.shape[0] == aux_attention_mask.shape[0]) 
        img_id=sample["img_id"]
        aux_id=sample["aux_idx"]
        pixel_values,aux_values,rcnn_values=loading(img_id,aux_id,mode=self.mode)

        span_num=bert_spans_tensor.shape[0]
        return main_tokens_tensor,aux_tokens_tensor, main_attention_mask,aux_attention_mask,bert_spans_tensor, spans_ner_label_tensor,span_num,attention_with_image,pixel_values,aux_values,rcnn_values,sample

    def __len__(self):
        return len(self.samples)

    def _get_input_tensors(self, tokens, spans, spans_ner_label):
        start2idx = []
        end2idx = []
        
        bert_tokens = []
        context_attention_mask=[]
        bert_tokens.append(self.tokenizer.cls_token)
        for token in tokens:
            start2idx.append(len(bert_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens)-1)
        bert_tokens.append(self.tokenizer.sep_token)
        context_attention_mask +=[1]*len(bert_tokens)

        pad_num=self.max_context_len-len(bert_tokens)
        bert_tokens += [self.tokenizer.pad_token]*pad_num
        context_attention_mask +=[0]*pad_num 

        aux_tokens=self.tokenizer.convert_tokens_to_ids(bert_tokens)
        aux_attention_mask=context_attention_mask.copy()

        question_tokens,question_attention_mask = self.appendquestion()

        bert_tokens.extend(question_tokens)

        if self.args.directional:
            attention_mask=self.directional_attention(context_attention_mask,question_attention_mask)
            # print("this exam use directional attention !!!")
        else:
            attention_mask=context_attention_mask+[1]*len(question_tokens)
            # print("this exam use directional attention !!!")

        attention_with_image=self.directional_attention_with_patch(context_attention_mask,question_attention_mask,61)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        main_tokens_tensor = torch.tensor(indexed_tokens)  #max_len

        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
        bert_spans_tensor = torch.tensor(bert_spans)                  #num,3

        spans_ner_label_tensor = torch.tensor(spans_ner_label)       #num
        main_attention_mask =torch.tensor(attention_mask)

        aux_tokens_tensor=torch.tensor(aux_tokens)
        aux_attention_mask=torch.tensor(aux_attention_mask)
        attention_with_image =torch.tensor(attention_with_image)
        return main_tokens_tensor,aux_tokens_tensor, main_attention_mask,aux_attention_mask,bert_spans_tensor, spans_ner_label_tensor,attention_with_image

    def appendquestion(self):
        question_tokens= []
        question_attention_mask = []
        if self.args.question:
            for index,question in enumerate( visual_entity_question,2):
                for token in question.split(" "):
                    sub_tokens = self.tokenizer.tokenize(token)
                    question_tokens += sub_tokens
                    question_attention_mask+=[index]*len(sub_tokens)

                question_tokens.append(self.tokenizer.sep_token)
                question_attention_mask+=[index]

                if not self.args.multiquestion:
                    return question_tokens,question_attention_mask  #one question

            return question_tokens,question_attention_mask    #multi question
        else:
            return question_tokens,question_attention_mask  #no question

    def directional_attention(self,context_attention_mask,question_attention_mask):
        temp = context_attention_mask+question_attention_mask
        attn=[]
        for index,cur_mask in enumerate(temp):
            token_atten=[]
            
            for index , mask_value in enumerate(temp):
                if cur_mask==1:
                    if mask_value!=0:
                        token_atten.append(1)
                    else:
                        token_atten.append(0)
                elif cur_mask==0:
                    token_atten.append(0)
                else:
                    if mask_value ==1 or mask_value==0:
                        token_atten.append(mask_value)
                    else:
                        token_atten.append(int((cur_mask/mask_value)==1))

            attn.append(token_atten)
        return attn

    def directional_attention_with_patch(self,context_attention_mask,question_attention_mask,patch_num):
        temp = context_attention_mask+question_attention_mask
        attn=[]
        #for text attention
        for index,cur_mask in enumerate(temp):
            token_atten=[]
            
            for index , mask_value in enumerate(temp):
                if cur_mask==1:
                    if mask_value!=0:
                        token_atten.append(1)
                    else:
                        token_atten.append(0)
                elif cur_mask==0:
                    token_atten.append(0)
                else:
                    if mask_value ==1 or mask_value==0:
                        token_atten.append(mask_value)
                    else:
                        token_atten.append(int((cur_mask/mask_value)==1))
            if cur_mask==0:
                token_atten+=[0]*patch_num
            else:
                token_atten+=[1]*patch_num

            attn.append(token_atten)

        image_temp=temp+ [1]*patch_num
        for patch in range(patch_num):
            attn.append(image_temp)
        return attn



def collate_fn(batch):
    batch_main_tokens_tensor           = [ data[0] for data in batch]
    batch_aux_tokens_tensor            = [ data[1] for data in batch]
    batch_main_attention_mask          = [ data[2] for data in batch]
    batch_aux_attention_mask           = [ data[3] for data in batch]

    batch_bert_spans_tensor         = [ data[4] for data in batch]
    batch_spans_ner_label_tensor    = [ data[5] for data in batch]

    batch_span_num                  = [ data[6] for data in batch]
    batch_attention_with_image      = [ data[7] for data in batch]

    batch_pixel_values              = [ data[-4] for data in batch]
    batch_aux_values                = [ data[-3] for data in batch]
    batch_rcnn_values               = [ data[-2] for data in batch]
    samples                          = [ data[-1] for data in batch]
    max_spans    =max(batch_span_num)

    new_bert_spans_tensor=[]
    new_spans_ner_label_tensor=[]
    spans_masks_tensor=[]

    for bert_spans_tensor, spans_ner_label_tensor in zip(batch_bert_spans_tensor, batch_spans_ner_label_tensor):
        # padding for spans
        num_spans = bert_spans_tensor.shape[0]
        spans_pad_length = max_spans - num_spans
        spans_mask_tensor = torch.ones(num_spans,dtype=torch.long)
        if spans_pad_length>0:
            pad = torch.full([spans_pad_length,bert_spans_tensor.shape[1]], 0, dtype=torch.long)
            bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=0)   #max,3

            mask_pad = torch.zeros(spans_pad_length,dtype=torch.long)  
            spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=-1)

            spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=-1)

            new_bert_spans_tensor.append(bert_spans_tensor)
            new_spans_ner_label_tensor.append(spans_ner_label_tensor)
            spans_masks_tensor.append(spans_mask_tensor)
        else:
            new_bert_spans_tensor.append(bert_spans_tensor)
            new_spans_ner_label_tensor.append(spans_ner_label_tensor)
            spans_masks_tensor.append(spans_mask_tensor)
    batch_main_tokens_tensor           = torch.stack(batch_main_tokens_tensor)
    batch_aux_tokens_tensor            = torch.stack(batch_aux_tokens_tensor)
    batch_main_attention_mask          = torch.stack(batch_main_attention_mask)
    batch_aux_attention_mask           = torch.stack(batch_aux_attention_mask)

    # batch_tokens_tensor             = torch.stack(batch_tokens_tensor)
    batch_bert_spans_tensor         = torch.stack(new_bert_spans_tensor)
    batch_spans_ner_label_tensor    = torch.stack(new_spans_ner_label_tensor)
    batch_attention_with_image      = torch.stack(batch_attention_with_image)
    spans_masks_tensor              = torch.stack(spans_masks_tensor)
    batch_pixel_values              = torch.stack(batch_pixel_values)
    batch_aux_values                = torch.stack(batch_aux_values)
    batch_rcnn_values               = torch.stack(batch_rcnn_values)

    return batch_main_tokens_tensor,batch_aux_tokens_tensor,batch_main_attention_mask,batch_aux_attention_mask,batch_bert_spans_tensor,  batch_spans_ner_label_tensor,batch_attention_with_image,spans_masks_tensor,batch_pixel_values,batch_aux_values,batch_rcnn_values,samples
