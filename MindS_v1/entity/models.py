import logging
import numpy as np
import mindspore as ms 
from shared.outputtotal.model import VisualEncoder
import mindspore.nn as nn 
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.nn import CrossEntropyLoss

logger = logging.getLogger('root')

class BertForEntity(nn.Cell):
    def __init__(self,  args ,num_ner_labels):
        super(BertForEntity,self).__init__()

        self.args=args
        self.vb=VisualEncoder()
        param_dict=ms.load_checkpoint("shared/outputtotal/model.ckpt")
        new_param_dict={}
        for k,v in param_dict.items():
            if "visualbertencoder_0" in k:
                k=k.replace(".visualbert",".bert")
            new_param_dict[k]=v

        new_new_param_dict={}
        for k,v in new_param_dict.items():
            new_k="vb."+k
            new_new_param_dict[new_k]=v
        param_not_load, _=ms.load_param_into_net(self.vb, new_new_param_dict)
        self.ner_dropout = nn.Dropout(p=0.3)
        self.ner_classifier = nn.SequentialCell(
            nn.Dense(768*2, 300),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Dense(300, 2)
        )
        self.loss_fct =CrossEntropyLoss()

    def construct(self, main_input_ids, main_attention_mask,token_type_ids,
                    spans, spans_mask, spans_ner_label, 
                    attention_with_image,
                    pixel_values,aux_values, rcnn_values,sample,mode):

        sequence_output = self.vb(input_ids=main_input_ids, attention_mask=main_attention_mask,token_type_ids=token_type_ids,attention_with_image=attention_with_image,pixel_values=pixel_values,aux_values=aux_values,rcnn_values=rcnn_values)

        spans = Tensor(spans)

        spans_start = ops.reshape(spans[:, :, 0],(spans.shape[0],-1))
        spans_start_embedding=ops.zeros((spans.shape[0],spans.shape[1],sequence_output.shape[-1]))
        for index,(hiddens,start) in enumerate(zip(sequence_output,spans_start)) :
            span=ops.index_select(hiddens,0,start)
            spans_start_embedding[index]=span

        spans_end = ops.reshape(spans[:, :,1],(spans.shape[0],-1))
        spans_end_embedding=ops.zeros((spans.shape[0],spans.shape[1],sequence_output.shape[-1]))
        for index,(hiddens,end) in enumerate(zip(sequence_output,spans_end)) :
            span=ops.index_select(hiddens,0,end)
            spans_end_embedding[index]=span

        spans_embedding = ops.cat((spans_start_embedding, spans_end_embedding), axis=-1)
        
        spans_embedding =self.ner_dropout(spans_embedding)
        logits = self.ner_classifier(spans_embedding)
 
        active_loss = spans_mask.view(-1) == 1
        active_logits = logits.view(-1, logits.shape[-1])
        active_labels = ops.where(
            active_loss, spans_ner_label.view(-1), Tensor(self.loss_fct.ignore_index)
        )
        active_labels=active_labels.astype(np.int32)
        loss = self.loss_fct(active_logits, active_labels)

        if mode[0] == 0 :
            return loss
        else:
            return loss,active_logits,active_labels