import torch
from torch import nn,Tensor,device
from transformers import VisualBertModel,CLIPVisionModel,CLIPModel,CLIPConfig,BertModel,VisualBertConfig,RobertaModel
from shared.ClipViT import CLIPVisionTransformer


class VisualEncoder(nn.Module):
    def __init__(self,args):
        super(VisualEncoder, self).__init__()
        self.args =args 
        #for text
        self.main_bert=RobertaModel.from_pretrained("roberta-base")
        self.aux_bert=RobertaModel.from_pretrained("roberta-base")  #unused
        
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
        
        #for auxilary text fusion 
        self.norm= nn.LayerNorm(768) #unused

    def forward(self,main_input_ids,main_attention_mask,
                     aux_input_ids,aux_attention_mask,attention_with_image,
                     pixel_values,aux_values, rcnn_values):
        
        main_text_embeds=self.main_bert(input_ids=main_input_ids,attention_mask=main_attention_mask,return_dict=True).last_hidden_state
        # aux_hidden =self.aux_bert(input_ids=aux_input_ids,attention_mask=aux_attention_mask,return_dict=True).last_hidden_state #unused
        visual_embeds=self.vit(pixel_values, aux_values, rcnn_values,return_dict=True).last_hidden_state
        fusion_embeds=torch.cat((main_text_embeds,visual_embeds),1)
        if self.args.directional:
            fusion_attention_mask=attention_with_image
        else:
            # visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).cuda()
            attention_with_image =attention_with_image.sum(-1)
            fusion_attention_mask =torch.where(attention_with_image>0, torch.ones_like(attention_with_image).cuda(),torch.zeros_like(attention_with_image).cuda()) 

        extended_attention_mask=self.get_extended_attention_mask(fusion_attention_mask)
        fusion_hidden = self.transformer(fusion_embeds,attention_mask=extended_attention_mask,return_dict=True).last_hidden_state

        return fusion_hidden

    #copied from transformers/src/transformers/modeling_utils.py
    def get_extended_attention_mask(
            self, attention_mask: Tensor, device: device = None
        ) -> Tensor:

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

