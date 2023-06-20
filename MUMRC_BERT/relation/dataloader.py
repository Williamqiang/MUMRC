import json
import os 
import sys
import torch
from torch.utils.data import  Dataset
from PIL import Image
import numpy as np
import math
import random

from shared.loadimg import loading
from shared.question_template import visual_relation_question as que
visual_relation_question=que

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, main_input_ids, main_attention_mask,aux_input_ids,aux_attention_mask,attention_with_image,label_id,sub_idx, obj_idx,img_id,aux_idx):
        self.main_input_ids = main_input_ids
        self.main_attention_mask = main_attention_mask
        self.aux_input_ids= aux_input_ids
        self.aux_attention_mask =aux_attention_mask
        self.attention_with_image = attention_with_image
        self.label_id = label_id
        self.sub_idx = sub_idx
        self.obj_idx = obj_idx
        self.img_id=img_id
        self.aux_idx=aux_idx


class MNREDateset(Dataset):
    def __init__(self,samples,tokenizer,label2id,mode,args=None):
        self.args=args
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_context_length=args.max_context_length
        self.mode = mode
        self.unused_tokens=True
        self.label2id=label2id
        self.SUBJECT_START   = "[unused%d]"%1
        self.SUBJECT_END     = "[unused%d]"%2
        self.OBJECT_START    = "[unused%d]"%3
        self.OBJECT_END      = "[unused%d]"%4

        self.SUBJECT_NER     = "[unused%d]"%5      #get_special_token("SUBJ=%s" % example['subj_type'])
        self.OBJECT_NER      = "[unused%d]"%6      #get_special_token("OBJ=%s" % example['obj_type'])

        self.SUBJECT_START_NER ="[unused%d]"%7                      #get_special_token("SUBJ_START=%s"%example['subj_type'])
        self.SUBJECT_END_NER   ="[unused%d]"%8                      #get_special_token("SUBJ_END=%s"%example['subj_type'])
        self.OBJECT_START_NER  ="[unused%d]"%9                      #get_special_token("OBJ_START=%s"%example['obj_type'])
        self.OBJECT_END_NER    ="[unused%d]"%10                     #get_special_token("OBJ_END=%s"%example['obj_type'])
    
    def __getitem__(self,index):
        sample=self.samples[index]

        feature= self.convert_examples_to_features(sample)

        img_id=sample["img_id"]
        aux_id=sample["aux_idx"]
        pixel_values,aux_values,rcnn_values=loading(img_id,aux_id,mode=self.mode)

        return feature,pixel_values,aux_values,rcnn_values,sample
    
    def __len__(self):
        return len(self.samples)

    def convert_examples_to_features(self,example):
        context_tokens = [self.tokenizer.cls_token]
        position_ids =list(range(self.max_context_length))
        for i, token in enumerate(example['token']):
            if i == example['subj_start']:
                sub_idx = len(context_tokens)
                context_tokens.append(self.SUBJECT_START_NER)
            if i == example['obj_start']:
                obj_idx = len(context_tokens)
                context_tokens.append(self.OBJECT_START_NER)

            for sub_token in self.tokenizer.tokenize(token):
                context_tokens.append(sub_token)

            if i == example['subj_end']:
                sub_end_idx=len(context_tokens)
                context_tokens.append(self.SUBJECT_END_NER)
            if i == example['obj_end']:
                obj_end_idx=len(context_tokens)
                context_tokens.append(self.OBJECT_END_NER)
        context_tokens.append(self.tokenizer.sep_token)
        context_attention_mask = [1]*len(context_tokens)

        assert (self.max_context_length-len(context_tokens))>=0

        pad_num=self.max_context_length-len(context_tokens)
        context_attention_mask += [0]*pad_num
        context_tokens += [self.tokenizer.pad_token]*pad_num
        aux_input_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
        aux_attention_mask= context_attention_mask.copy()

        question_tokens,question_attention_mask=self.appendquestion(sub_idx,sub_end_idx,obj_idx,obj_end_idx)

        if self.args.directional:
            main_attention_mask=self.directional_attention(context_attention_mask,question_attention_mask)
        else:
            main_attention_mask=context_attention_mask+[1]*len(question_tokens)

        attention_with_image=self.directional_attention_with_patch(context_attention_mask,question_attention_mask,61)
        
        main_input_tokens = context_tokens + question_tokens
        main_input_ids = self.tokenizer.convert_tokens_to_ids(main_input_tokens)
        label_id = self.label2id[example['relation']]

        assert len(main_input_ids) == (self.max_context_length + len(question_tokens))
        assert len(main_attention_mask) == (self.max_context_length + len(question_tokens))

        feature=InputFeatures(main_input_ids=main_input_ids,
                             main_attention_mask=main_attention_mask,
                             aux_input_ids= aux_input_ids,
                             aux_attention_mask= aux_attention_mask,
                             attention_with_image=attention_with_image,
                                label_id=label_id,
                                sub_idx=sub_idx,
                                obj_idx=obj_idx,
                                img_id=example["img_id"],
                                aux_idx=example["aux_idx"])
        return feature

    def appendquestion(self,sub_idx,sub_end_idx,obj_idx,obj_end_idx):
        question_tokens= []
        question_attention_mask = []
        question_position_ids = []
        position_count=self.max_context_length

        if self.args.question:
            for index,question in enumerate( visual_relation_question,2):
                for token in question.split(" "):
                    temp=[]
                    if token=="{":
                        question_tokens.append(self.SUBJECT_START_NER)
                        question_attention_mask+=[index]
                        question_position_ids.append(sub_idx)
                    elif token=="}":
                        question_tokens.append(self.SUBJECT_END_NER)
                        question_attention_mask+=[index]
                        question_position_ids.append(sub_end_idx)
                    elif token=="[":
                        question_tokens.append(self.OBJECT_START_NER)
                        question_attention_mask+=[index]
                        question_position_ids.append(obj_idx)
                    elif token=="]":
                        question_tokens.append(self.OBJECT_END_NER)
                        question_attention_mask+=[index]
                        question_position_ids.append(obj_end_idx)
                    else:
                        sub_tokens=self.tokenizer.tokenize(token)
                        question_tokens.extend(sub_tokens)
                        question_attention_mask+=[index]*len(sub_tokens)
                        question_position_ids += list(range(position_count,position_count+len(sub_tokens)))
                        position_count +=len(sub_tokens)

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
    batch_features                  = [ data[0] for data in batch ]

    batch_pixel_values              = [ data[-4] for data in batch]
    batch_aux_values                = [ data[-3] for data in batch]
    batch_rcnn_values               = [ data[-2] for data in batch]
    samples                         = [ data[-1] for data in batch]
   
    batch_main_input_ids      = [torch.tensor(feature.main_input_ids)  for feature in batch_features]
    batch_main_attention_mask = [torch.tensor(feature.main_attention_mask) for feature in batch_features]
    batch_aux_input_ids       = [torch.tensor(feature.aux_input_ids)  for feature in batch_features]
    batch_aux_attention_mask  = [torch.tensor(feature.aux_attention_mask) for feature in batch_features]
    batch_attention_with_image= [torch.tensor(feature.attention_with_image) for feature in batch_features]
    batch_label_ids           = [torch.tensor(feature.label_id )  for feature in batch_features]
    batch_sub_ids             = [torch.tensor(feature.sub_idx)    for feature in batch_features]
    batch_obj_ids             = [torch.tensor(feature.obj_idx )   for feature in batch_features]


    # batch_tokens_tensor             = torch.stack(batch_tokens_tensor)
    batch_main_input_ids            =torch.stack(batch_main_input_ids)
    batch_main_attention_mask       =torch.stack(batch_main_attention_mask)
    batch_aux_input_ids             =torch.stack(batch_aux_input_ids)
    batch_aux_attention_mask        =torch.stack(batch_aux_attention_mask)
    batch_attention_with_image      =torch.stack(batch_attention_with_image)
    batch_label_ids                 =torch.stack(batch_label_ids)
    batch_sub_ids                   =torch.stack(batch_sub_ids)
    batch_obj_ids                   =torch.stack(batch_obj_ids)

    batch_pixel_values              = torch.stack(batch_pixel_values)
    batch_aux_values                = torch.stack(batch_aux_values)
    batch_rcnn_values               = torch.stack(batch_rcnn_values)
    return batch_main_input_ids,batch_main_attention_mask,batch_aux_input_ids,batch_aux_attention_mask,batch_attention_with_image,batch_sub_ids,batch_obj_ids,batch_label_ids,batch_pixel_values,batch_aux_values,batch_rcnn_values,samples
