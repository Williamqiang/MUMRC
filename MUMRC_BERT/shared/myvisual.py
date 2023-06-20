import torch
from torch import nn,Tensor,device
from transformers import VisualBertModel,CLIPVisionModel,CLIPModel,CLIPConfig,BertModel,VisualBertConfig
from shared.ClipViT import CLIPVisionTransformer
from shared.third import auxfusion,identity ,SupConLoss
# from allennlp.modules.transformer.transformer_stack import TransformerStack as Transformer


class VisualEncoder(nn.Module):
    def __init__(self,args):
        super(VisualEncoder, self).__init__()
        self.args =args 
        #for text
        self.main_bert=BertModel.from_pretrained("bert-base-uncased")
        self.aux_bert=BertModel.from_pretrained("bert-base-uncased") #unused
        
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

# class OpenaiEncoder(nn.Module):
#     def __init__(self,args):
#         super(OpenaiEncoder, self).__init__()
#         print("这里是ner的OpenaiEncoder模型")
#         self.args =args 
#         #for text
#         self.main_bert=BertModel.from_pretrained("PretrainModel/bert-base-uncased")
#         self.aux_bert=BertModel.from_pretrained("PretrainModel/bert-base-uncased")
        
#         #for image 
#         config=CLIPConfig.from_pretrained("openai/clip-vit-base-patch32").vision_config
#         state_dict= CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model.state_dict()
#         self.vit=CLIPVisionTransformer(config)
#         self.vit.load_state_dict(state_dict,strict=False)
        
#         #for modality fusion 
#         self.hidden_size=768
#         self.transformer = Transformer(
#                                 num_hidden_layers=self.args.fusion_num_hidden_layers,
#                                 hidden_size=self.hidden_size,
#                                 intermediate_size=self.hidden_size*4,
#                                 num_attention_heads =self.args.fusion_attention_head,
#                                 attention_dropout=self.args.fusion_attention_drop,
#                                 hidden_dropout = self.args.fusion_hidden_drop,
#                                 activation ="gelu"
#                                 )
        
#         #for auxilary text fusion
#         self.norm= nn.LayerNorm(768)
#         self.auxliary = nn.Linear(768,768)
#     def forward(self,main_input_ids,main_attention_mask,
#                      aux_input_ids,aux_attention_mask,attention_with_image,
#                      pixel_values,aux_values, rcnn_values):
#         main_text_embeds=self.main_bert(input_ids=main_input_ids,attention_mask=main_attention_mask,return_dict=True).last_hidden_state
#         aux_hidden =self.aux_bert(input_ids=aux_input_ids,attention_mask=aux_attention_mask,return_dict=True).last_hidden_state
        
#         to_aux_mask_main= main_attention_mask.sum(-1,keepdim=True)
#         to_aux_mask_aux = aux_attention_mask.unsqueeze(1) 

#         mask = to_aux_mask_main * to_aux_mask_aux
#         mask = torch.where(mask>0,torch.zeros_like(mask).cuda(),-10000000 )

#         dot_matrix = torch.matmul(main_text_embeds,aux_hidden.transpose(1,2)) #B,m,n
#         dot_matrix =dot_matrix + mask

#         cosine_score = nn.Softmax(-1)(dot_matrix) #B,m,n
#         useful_info  = torch.matmul(cosine_score,aux_hidden) #b,m,H
#         useful_info =   self.auxliary(useful_info)
#         main_text_embeds  = main_text_embeds + useful_info


#         visual_embeds=self.vit(pixel_values, aux_values, rcnn_values,return_dict=True).last_hidden_state
#         # main_text_embeds=auxfusion(main_text_embeds,aux_text_embeds)
#         fusion_embeds=torch.cat((main_text_embeds,visual_embeds),1)

#         if self.args.directional:
#             # bsz,patch_num=visual_embeds.shape[:-1]
#             # _,seq_len= main_input_ids.shape
#             # visual_template= main_attention_mask[:,0,:].unsqueeze(1).repeat(1,patch_num,1)
#             # right_attention= torch.ones(bsz,patch_num+seq_len,patch_num).cuda()
#             # fusion_attention_mask= torch.cat((main_attention_mask,visual_template),1)
#             # fusion_attention_mask=torch.cat((fusion_attention_mask,right_attention),-1)
#             fusion_attention_mask=attention_with_image
#         else:
#             attention_with_image =attention_with_image.sum(-1)
#             fusion_attention_mask =torch.where(attention_with_image>0, torch.ones_like(attention_with_image).cuda(),torch.zeros_like(attention_with_image).cuda()) 
#             # visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).cuda()
#             # fusion_attention_mask = torch.cat((main_attention_mask,visual_attention_mask),1)

#         fusion_hidden = self.transformer(fusion_embeds,attention_mask=fusion_attention_mask).final_hidden_states

#         # to_aux_mask_main= attention_with_image.sum(-1,keepdim=True)
#         # to_aux_mask_aux = aux_attention_mask.unsqueeze(1) 

#         # mask = to_aux_mask_main * to_aux_mask_aux
#         # mask = torch.where(mask>0,torch.zeros_like(mask).cuda(),-10000000 )

#         # dot_matrix = torch.matmul(fusion_hidden,aux_hidden.transpose(1,2)) #B,m,n
#         # dot_matrix =dot_matrix + mask

#         # cosine_score = nn.Softmax(-1)(dot_matrix) #B,m,n
#         # useful_info  = torch.matmul(cosine_score,aux_hidden) #b,m,H
#         # useful_info =   self.auxliary(useful_info)
#         # fusion_hidden  = fusion_hidden + useful_info

#         # fusion_hidden =self.norm(fusion_hidden)

#         return fusion_hidden
