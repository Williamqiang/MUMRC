import json
import os 
import sys
from PIL import Image
import numpy as np
import math
import random

from shared.loadimg import loading
from shared.question_template import visual_relation_question as que
visual_relation_question=que

class MNREDateset:
    def __init__(self,samples,tokenizer,label2id,mode,args=None):
        self.args=args
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_context_length=args.max_context_length
        self.mode = mode
        self.mode_id = 0 if mode=="train" else 1

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

        main_input_ids,main_attention_mask,token_type_id,attention_with_image,label_id,sub_idx,obj_idx,img_id,aux_idx= self.convert_examples_to_features(sample)

        img_id=sample["img_id"]
        aux_id=sample["aux_idx"]
        pixel_values,aux_values,rcnn_values=loading(img_id,aux_id,mode=self.mode)

        return main_input_ids,main_attention_mask,token_type_id,attention_with_image,label_id,sub_idx,obj_idx,pixel_values,aux_values,rcnn_values,self.mode_id
    
    def __len__(self):
        return len(self.samples)

    def get_samples(self):
        return self.samples
        
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


        question_tokens,question_attention_mask=self.appendquestion(sub_idx,sub_end_idx,obj_idx,obj_end_idx)

        if self.args.directional:
            main_attention_mask=self.directional_attention(context_attention_mask,question_attention_mask)


        attention_with_image=self.directional_attention_with_patch(context_attention_mask,question_attention_mask,61)
        
        main_input_tokens = context_tokens + question_tokens
        main_input_ids = self.tokenizer.convert_tokens_to_ids(main_input_tokens)
        token_type_id = np.zeros_like(main_input_ids)
        label_id = self.label2id[example['relation']]

        assert len(main_input_ids) == (self.max_context_length + len(question_tokens))
        assert len(main_attention_mask) == (self.max_context_length + len(question_tokens))

        return main_input_ids,main_attention_mask,token_type_id,attention_with_image,label_id,sub_idx,obj_idx,example["img_id"],example["aux_idx"]
        # return feature

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

def collate_fn(batch_main_input_ids,batch_main_attention_mask,batch_token_type_id,batch_attention_with_image,batch_label_ids,batch_sub_ids,batch_obj_ids,batch_pixel_values,batch_aux_values,batch_rcnn_values,mode_id,BatchInfo):


    # batch_features                  = [ data[0] for data in batch ]

    # batch_main_input_ids      = [ data[0] for data in batch]
    # batch_main_attention_mask = [ data[1] for data in batch]
    # batch_token_type_id       = [ data[2] for data in batch] 
    # batch_attention_with_image= [ data[3] for data in batch] 
    # batch_label_ids           = [ data[4] for data in batch] 
    # batch_sub_ids             = [ data[5] for data in batch] 
    # batch_obj_ids             = [ data[6] for data in batch] 

    # batch_pixel_values              = [ data[-5] for data in batch]
    # batch_aux_values                = [ data[-4] for data in batch]
    # batch_rcnn_values               = [ data[-3] for data in batch]
    # samples                         = [ data[-2] for data in batch]
    # mode_id                         = [ data[-1] for data in batch]



    input_ids            =np.stack(batch_main_input_ids)
    attention_mask       =np.stack(batch_main_attention_mask)
    token_type_id       = np.stack(batch_token_type_id)

    attention_with_image      =np.stack(batch_attention_with_image)
    label_ids                 =np.stack(batch_label_ids)
    sub_ids                   =np.stack(batch_sub_ids)
    obj_ids                   =np.stack(batch_obj_ids)

    pixel_values              = np.stack(batch_pixel_values)
    aux_values                = np.stack(batch_aux_values)
    rcnn_values               = np.stack(batch_rcnn_values)
    mode_id                   = np.stack(mode_id)

    # print("input_ids.shape",input_ids.shape)
    # print("attention_mask.shape",attention_mask.shape)
    # print("token_type_id.shape",token_type_id.shape)

    # print("attention_with_image.shape",attention_with_image.shape)
    # print("pixel_values.shape",pixel_values.shape)
    # print("aux_values.shape",aux_values.shape)
    # print("rcnn_values.shape",rcnn_values.shape)

    # print("label_ids.shape",label_ids.shape)
    # print("sub_ids.shape",sub_ids.shape)
    # print("obj_ids.shape",obj_ids.shape)
    return input_ids,attention_mask,token_type_id,attention_with_image,label_ids,sub_ids,obj_ids,pixel_values,aux_values,rcnn_values,mode_id
