# import torch
# from torch import nn,Tensor,device
from transformers import VisualBertModel,CLIPVisionModel,CLIPModel,CLIPConfig,BertModel,VisualBertConfig
from shared.ClipViT import CLIPVisionTransformer

import mindspore.nn as nn 
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.nn import CrossEntropyLoss

class VisualEncoder(nn.Module):
    def __init__(self,args):
        super(VisualEncoder, self).__init__()
        self.args =args 
        #for text
        self.bert=BertModel.from_pretrained("bert-base-uncased")
        
        #for image 
        config=CLIPConfig.from_pretrained("openai/clip-vit-base-patch32").vision_config
        state_dict= CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model.state_dict()
        self.vit=CLIPVisionTransformer(config)
        self.vit.load_state_dict(state_dict,strict=False)

        #for modality fusion 
        self.visualbert_config=VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        self.visualbert_config.attention_probs_dropout_prob=self.args.fusion_attention_drop
        self.visualbert_config.hidden_dropout_prob=self.args.fusion_hidden_drop
        self.visualbert_config.num_attention_heads=self.args.fusion_attention_head
        self.visualbert_config.hidden_act=self.args.hidden_act
        self.transformer = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre",config=self.visualbert_config).encoder
        
        self.norm= nn.LayerNorm(768) #unused

    def forward(self,main_input_ids,main_attention_mask,
                     attention_with_image,
                     pixel_values,aux_values, rcnn_values):
        
        main_text_embeds=self.bert(input_ids=main_input_ids,attention_mask=main_attention_mask,return_dict=True).last_hidden_state
        visual_embeds=self.vit(pixel_values, aux_values, rcnn_values,return_dict=True).last_hidden_state
        fusion_embeds=ops.cat((main_text_embeds,visual_embeds),1)
        if self.args.directional:
            fusion_attention_mask=attention_with_image

        extended_attention_mask=self.get_extended_attention_mask(fusion_attention_mask)
        fusion_hidden = self.transformer(fusion_embeds,attention_mask=extended_attention_mask,return_dict=True).last_hidden_state

        return fusion_hidden

    #copied from transformers/src/transformers/modeling_utils.py
    def get_extended_attention_mask(self, attention_mask) :

            if attention_mask.dim() == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    f"Wrong shape for attention_mask (shape {attention_mask.shape})"
                )

            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            return extended_attention_mask

