import json
import os 
import sys
from PIL import Image
import numpy as np
import math
import random
from shared.loadimg import loading
from shared.question_template import visual_entity_question as que 
visual_entity_question=que

class MNREDateset:
    def __init__(self,samples,tokenizer,mode,args=None):
        self.args=args
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_context_len=args.max_context_length
        self.mode = mode
        self.mode_id = 0 if mode=="train" else 1
        # self.mode_id = 0 if mode=="train" or mode=="test" else 1

    def __getitem__(self,index):
        sample=self.samples[index]
        tokens = sample['tokens']
        spans = sample['spans']
        spans_ner_label = sample['spans_label']
        input_id, attention_mask,spans, spans_ner_label ,attention_with_image= self._get_input_tensors(tokens, spans, spans_ner_label)
        token_type_id = np.zeros_like(input_id)
        assert(input_id.shape[0] == attention_mask.shape[0]) 
        assert(spans.shape[0] == spans_ner_label.shape[0]) 
        img_id=sample["img_id"]
        aux_id=sample["aux_idx"]
        pixel_values,aux_values,rcnn_values=loading(img_id,aux_id,mode=self.mode)

        span_num=spans.shape[0]
        return input_id, attention_mask,token_type_id,spans, spans_ner_label,span_num,attention_with_image,pixel_values,aux_values,rcnn_values,sample,self.mode_id

    def __len__(self):
        return len(self.samples)

    def get_samples(self):
        return self.samples
        
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

        question_tokens,question_attention_mask = self.appendquestion()

        bert_tokens.extend(question_tokens)

        if self.args.directional:
            attention_mask=self.directional_attention(context_attention_mask,question_attention_mask)


        attention_with_image=self.directional_attention_with_patch(context_attention_mask,question_attention_mask,61)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        input_id = np.array(indexed_tokens)  #max_len

        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
        spans = np.array(bert_spans)                  #num,3

        ner_label = np.array(spans_ner_label)       #num
        attention_mask =np.array(attention_mask)

        attention_with_image =np.array(attention_with_image)
        return input_id, attention_mask,spans, ner_label,attention_with_image

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



def collate_fn(input_id, attention_mask,token_type_id,spans, spans_ner_label,span_num,attention_with_image,pixel_values,aux_values,rcnn_values,sample,mode_id,BatchInfo):
    # print("#"*50)
    # print(batch)
    # print(len(input_id))
    # exit()

    # print(sample)
    # exit()
    max_spans    =max(span_num)

    new_bert_spans_tensor=[]
    new_spans_ner_label_tensor=[]
    spans_masks_tensor=[]

    for bert_spans_tensor, spans_ner_label_tensor in zip(spans, spans_ner_label):
        # padding for spans
        num_spans = bert_spans_tensor.shape[0]
        spans_pad_length = max_spans - num_spans
        spans_mask_tensor = np.ones(num_spans,dtype=np.long)
        if spans_pad_length>0:
            pad = np.full([spans_pad_length,bert_spans_tensor.shape[1]], 0, dtype=np.long)
            bert_spans_tensor = np.concatenate((bert_spans_tensor, pad), axis=0)   #max,3

            mask_pad = np.zeros(spans_pad_length,dtype=np.long)  
            spans_mask_tensor = np.concatenate((spans_mask_tensor, mask_pad), axis=-1)

            spans_ner_label_tensor = np.concatenate((spans_ner_label_tensor, mask_pad), axis=-1)

            new_bert_spans_tensor.append(bert_spans_tensor)
            new_spans_ner_label_tensor.append(spans_ner_label_tensor)
            spans_masks_tensor.append(spans_mask_tensor)
        else:
            new_bert_spans_tensor.append(bert_spans_tensor)
            new_spans_ner_label_tensor.append(spans_ner_label_tensor)
            spans_masks_tensor.append(spans_mask_tensor)

    input_id           = np.stack(input_id)
    attention_mask          = np.stack(attention_mask)

    spans         = np.stack(new_bert_spans_tensor)
    spans_ner_label    = np.stack(new_spans_ner_label_tensor)
    attention_with_image      = np.stack(attention_with_image)
    spans_masks_tensor              = np.stack(spans_masks_tensor)
    pixel_values              = np.stack(pixel_values)
    aux_values                = np.stack(aux_values)
    rcnn_values               = np.stack(rcnn_values)
    token_type_id               = np.stack(token_type_id)
    mode_id = np.stack(mode_id)

    # print("input_id.shape",input_id.shape)
    # print("attention_mask.shape",attention_mask.shape)
    # print("token_type_id.shape",token_type_id.shape)

    # print("attention_with_image.shape",attention_with_image.shape)
    # print("pixel_values.shape",pixel_values.shape)
    # print("aux_values.shape",aux_values.shape)
    # print("rcnn_values.shape",rcnn_values.shape)
    
    # print("mode_id.shape",mode_id.shape)
    # print("sub_ids.shape",sub_ids.shape)
    # print("obj_ids.shape",obj_ids.shape)
    return input_id, attention_mask, token_type_id, spans,  spans_masks_tensor,spans_ner_label,attention_with_image,pixel_values,aux_values,rcnn_values,sample,mode_id
