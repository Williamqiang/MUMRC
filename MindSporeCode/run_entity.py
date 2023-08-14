import json
import argparse
import os
import sys
import random
import logging
import time
from tqdm import tqdm
import numpy as np

from shared.data_structures import Dataset
from shared.const import task_ner_labels, get_labelmap
from entity.utils import convert_dataset_to_samples, NpEncoder
from entity.models import BertForEntity
from entity.dataloader import MNREDateset,collate_fn
from transformers import BertTokenizer
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn 
from mindspore.train import Model,LossMonitor, TimeMonitor,Metric
from entity.callback import Traincallback,Evalcallback
from mindspore.nn import CrossEntropyLoss
import numpy as np

ms.set_context(device_target='GPU', device_id=0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')

class PRF(Metric):
    def __init__(self, tot_gold, eval_type='MUMRC_PRF'):
        super(PRF, self).__init__()
        # self.tot_gold=0
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""

        self.l_cor = 0
        self.l_tot = 0 
        self.cor =0
        self.tot_pred=0
        self.tot_gold=0

    def update(self,active_logits,active_labels):
        __, predicted_label = active_logits.max(1,return_indices=True)
        for gold,pred in zip(active_labels,predicted_label):
            if gold>=0:
                self.l_tot += 1
            else:
                continue

            if pred == gold:
                self.l_cor += 1
            if pred != 0 and gold != 0 and pred == gold:
                self.cor += 1
            if pred != 0:
                self.tot_pred += 1
            
            if gold !=0 and gold !=-100:
                self.tot_gold += 1
                   
    def eval(self):
        acc = self.l_cor / self.l_tot
        print('Cor: %d, Pred TOT: %d, Gold TOT: %d'%(self.cor, self.tot_pred, self.tot_gold))
        p = self.cor / self.tot_pred if self.cor > 0 else 0.0
        r = self.cor / self.tot_gold if self.cor > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if self.cor > 0 else 0.0
        print('Acc: %.5f P: %.5f, R: %.5f, F1: %.5f'%(acc,p, r, f1))

        return (p,r,f1)

class Output_ner(object):
    def __init__(self,net,data,dataset,mner_dataset,path,ner_id2label):
        super().__init__()
        self.net = net
        self.data = data 
        self.dataset = dataset
        self.samples = mner_dataset.get_samples()
        self.path = path
        self.ner_id2label = ner_id2label

    def eval(self):
        # net =net1
        self.net.set_train(False)
        ner_result = {}
        tot_pred_ett = 0
        for i,(input_id, attention_mask,token_type_id,spans, spans_masks,spans_ner_label,attention_with_image,pixel_values,aux_values,rcnn_values,samples,mode_id) in enumerate(self.dataset):
            loss,logit,label = self.net(input_id,attention_mask,token_type_id,spans,spans_masks,spans_ner_label,attention_with_image,pixel_values,aux_values,rcnn_values,samples,mode_id)
            __, predicted_label = logit.max(1,return_indices=True)

            samples= self.samples[i*16:i*16+16]
            lenght = len(samples)
            predicted_label = predicted_label.reshape(lenght,-1).asnumpy()
            predicted = []
            for i, sample in enumerate(samples):
                ner = []
                for j in range(len(sample['spans'])):
                    ner.append(predicted_label[i][j])
                predicted.append(ner)
                
            pred_ner = predicted
            for sample, preds in zip(samples, pred_ner):
                off = sample['sent_start_in_doc'] - sample['sent_start']
                k = sample['doc_key'] + '-' + str(sample['sentence_ix'])
                ner_result[k] = []
                for span, pred in zip(sample['spans'], preds):
                    span_id = '%s::%d::(%d,%d)'%(sample['doc_key'], sample['sentence_ix'], span[0]+off, span[1]+off)
                    if pred == 0:
                        continue
                    ner_result[k].append([span[0]+off, span[1]+off, ner_id2label[pred]])
                tot_pred_ett += len(ner_result[k])

        print('Total pred entities: %d'%tot_pred_ett)

        js = self.data.js
        for i, doc in enumerate(js):
            doc["predicted_ner"] = []
            doc["predicted_relations"] = []
            for j in range(len(doc["sentences"])):
                k = doc['doc_key'] + '-' + str(j)
                if k in ner_result:
                    doc["predicted_ner"].append(ner_result[k])
                else:
                    logger.info('%s not in NER results!'%k)
                    doc["predicted_ner"].append([])
                doc["predicted_relations"].append([])
            js[i] = doc

        print('Output predictions to %s..'%(self.path))
        with open(self.path, 'w') as f:
            f.write('\n'.join(json.dumps(doc, cls=NpEncoder) for doc in js))            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mnre')
    parser.add_argument('--data_dir', type=str, default=None, required=True,help="path to the preprocessed dataset")
    parser.add_argument('--output_dir', type=str, default='entity_output', help="output directory of the entity model")
    parser.add_argument('--max_span_length', type=int, default=8,  help="spans w/ length up to max_span_length are considered as candidates")
    parser.add_argument('--max_context_length', type=int, default=60)
    parser.add_argument('--train_batch_size', type=int, default=32, help="batch size during training")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="learning rate for the BERT encoder")
    parser.add_argument('--num_epoch', type=int, default=30, help="number of the training epochs")
    parser.add_argument('--do_train', action='store_true',  help="whether to run training")
    parser.add_argument('--do_eval', action='store_true',  help="whether to run evaluation")
    parser.add_argument('--eval_test', action='store_true', help="whether to evaluate on test set")
    parser.add_argument('--dev_pred_filename', type=str, default="ent_pred_dev.json", help="the prediction filename for the dev set")
    parser.add_argument('--test_pred_filename', type=str, default="ent_pred_test.json", help="the prediction filename for the test set")
    parser.add_argument('--question', action='store_true')
    parser.add_argument("--multiquestion",action='store_true')
    parser.add_argument("--directional",action='store_true')
    args = parser.parse_args()
    args.train_data = os.path.join(args.data_dir, 'train.json')
    args.dev_data = os.path.join(args.data_dir, 'dev.json')
    args.test_data = os.path.join(args.data_dir, 'test.json')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(sys.argv)
    logger.info(args)
    
    ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.task])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    num_ner_labels = len(task_ner_labels[args.task]) + 1
    net = BertForEntity(args, num_ner_labels=num_ner_labels)
    print("*"*50)

    column_names=["input_id", "attention_mask","token_type_id","spans", "spans_ner_label","span_num","attention_with_image","pixel_values","aux_values","rcnn_values","sample","mode_id"]
    dev_data = Dataset(args.dev_data)
    dev_samples, dev_ner = convert_dataset_to_samples(dev_data, args.max_span_length, ner_label2id=ner_label2id)
    dev_dataset =  MNREDateset(dev_samples,tokenizer,"dev",args=args)
    dev_ms_dataset = ds.GeneratorDataset(source=dev_dataset,column_names=column_names,shuffle= False)
    dev_ms_dataset = dev_ms_dataset.batch(batch_size = args.train_batch_size,per_batch_map=collate_fn)

    test_data = Dataset(args.test_data)
    test_samples, test_ner = convert_dataset_to_samples(test_data, args.max_span_length, ner_label2id=ner_label2id)
    test_dataset =  MNREDateset(test_samples,tokenizer,"test",args=args)
    test_ms_dataset = ds.GeneratorDataset(source=test_dataset,column_names=column_names,shuffle= False)
    test_ms_dataset = test_ms_dataset.batch(batch_size = args.train_batch_size,per_batch_map=collate_fn)

    dev_prediction_file = os.path.join(args.output_dir, args.dev_pred_filename)
    dev_out = Output_ner(net,dev_data,dev_ms_dataset,dev_dataset,dev_prediction_file,ner_id2label)

    test_prediction_file = os.path.join(args.output_dir, args.test_pred_filename)
    test_out = Output_ner(net,test_data,test_ms_dataset,test_dataset,test_prediction_file,ner_id2label)

    if args.do_train:
        train_data = Dataset(args.train_data)
        train_samples, train_ner = convert_dataset_to_samples(train_data, args.max_span_length, ner_label2id=ner_label2id)
        train_dataset =  MNREDateset(train_samples,tokenizer,"train",args=args)
        train_ms_dataset=ds.GeneratorDataset(source=train_dataset,column_names=column_names,shuffle= True)
        train_ms_dataset = train_ms_dataset.batch(batch_size = args.train_batch_size,per_batch_map=collate_fn)

        trainable_param = net.trainable_params()
        train_step_size = train_ms_dataset.get_dataset_size()
        lr = nn.warmup_lr(learning_rate=args.learning_rate,total_step=train_step_size * args.num_epoch,step_per_epoch=train_step_size,warmup_epoch=1)
        optimizer = nn.AdamWeightDecay(params=net.trainable_params(), learning_rate=lr,weight_decay=0.01)
        model = Model(net, optimizer=optimizer,eval_network=net,eval_indexes =[0,1,2],metrics={"PRF":PRF(dev_ner)})
        
        hist = {'loss':[], 'p':[], 'r':[], 'f':[]} # 训练过程记录
        train_cb=Traincallback(hist["loss"])
        eval_cb= Evalcallback(args,model,net,hist['p'],hist['r'],hist['f'],dev_ms_dataset,test_ms_dataset,dev_out,test_out)
        model.train(args.num_epoch,train_ms_dataset,callbacks=[train_cb,eval_cb,LossMonitor(100),TimeMonitor(0)])
