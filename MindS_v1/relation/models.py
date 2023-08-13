import numpy as np 
import mindspore as ms
import mindspore.nn as nn 
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.nn import CrossEntropyLoss

from shared.outputtotal.model import VisualEncoder

class BertForRelation(nn.Cell):
    def __init__(self, args, num_rel_labels):
        super(BertForRelation, self).__init__()
        self.args = args 
        self.num_labels = num_rel_labels
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

        self.re_dropout = nn.Dropout(p=0.1)
        self.re_classifier = nn.Dense(768 * 2, self.num_labels)
        self.loss_fct =CrossEntropyLoss()

    def construct(self, input_ids, attention_mask,token_type_id,attention_with_image,label_ids, 
                sub_ids, obj_ids,pixel_values,aux_values,rcnn_values,mode_id):

        sequence_output = self.vb(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_id,
                                attention_with_image=attention_with_image,
                                pixel_values=pixel_values,aux_values=aux_values,rcnn_values=rcnn_values)

        sub_output = ops.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_ids)])
        obj_output = ops.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_ids)])
        rep = ops.cat((sub_output, obj_output), axis=-1)
        rep = self.re_dropout(rep)
        logits = self.re_classifier(rep)
        labels=label_ids.astype(np.int32)

        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if mode_id[0] == 0 :
            return loss
        else:
            return loss,logits,labels