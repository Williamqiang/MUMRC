"""
This code is based on the file in SpanBERT repo: https://github.com/facebookresearch/SpanBERT/blob/master/code/run_tacred.py
"""

import argparse
import logging
import os
import random
import time
import json
import sys

import numpy as np
from tqdm import tqdm

from relation.models import BertForRelation
from entity.models import BertForEntity

from transformers import BertTokenizer

from relation.utils import generate_relation_data
from shared.const import task_rel_labels, task_ner_labels
from relation.dataloader import MNREDateset,collate_fn

import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn 
from mindspore.train import Model,LossMonitor, TimeMonitor,Metric
from relation.callback import Traincallback,Evalcallback
import faulthandler
faulthandler.enable()

ms.set_context(device_target='GPU',device_id=0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class PRF(Metric):
    def __init__(self):
        super(PRF, self).__init__()
        self.e2e_ngold=624
        self.clear()

    def clear(self):
        self.n_pred = 0
        self.n_gold = 0 
        self.n_correct =0

    def update(self,active_logits,active_labels):
        __, predicted_label = active_logits.max(1,return_indices=True)
        for label,pred in zip(active_labels,predicted_label):
            if pred != 0:
                self.n_pred += 1
            if label != 0:
                self.n_gold += 1
            if (pred != 0) and (label != 0) and (pred == label):
                self.n_correct += 1

    def eval(self):
        if self.n_correct == 0:
            print({'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
            return (0,0,0)
        else:
            prec = self.n_correct * 1.0 / self.n_pred
            recall = self.n_correct * 1.0 / self.n_gold
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            else:
                f1 = 0.0

            if self.e2e_ngold is not None:
                e2e_recall = self.n_correct * 1.0 / self.e2e_ngold
                e2e_f1 = 2.0 * prec * e2e_recall / (prec + e2e_recall)
            else:
                e2e_recall = e2e_f1 = 0.0

        
            res = {'precision': prec, 'recall': e2e_recall, 'f1': e2e_f1, 'task_recall': recall, 'task_f1': f1,  'n_correct': self.n_correct, 'n_pred': self.n_pred, 'n_gold': self.e2e_ngold, 'task_ngold': self.n_gold}
            for k,v in res.items():
                print(k,":",v)
            return (res["precision"],res["recall"],res["f1"])

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # get label_list
    if os.path.exists(os.path.join(args.output_dir, 'label_list.json')):
        with open(os.path.join(args.output_dir, 'label_list.json'), 'r') as f:
            label_list = json.load(f)
    else:
        label_list = [args.negative_label] + task_rel_labels[args.task]
        with open(os.path.join(args.output_dir, 'label_list.json'), 'w') as f:
            json.dump(label_list, f)

    print(label_list)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    column_names=["input_ids","attention_mask","token_type_id","attention_with_image","label_ids","sub_ids","obj_ids","pixel_values","aux_values","rcnn_values","mode_id"]
    # train set
    train_dataset, train_examples, train_nrel = generate_relation_data(args.train_file, use_gold=True)
    train_data =MNREDateset(train_examples,tokenizer,label2id,"train",args)
    train_ms_dataset=ds.GeneratorDataset(source=train_data,column_names=column_names,shuffle= True)
    train_ms_dataset = train_ms_dataset.batch(batch_size = args.train_batch_size,per_batch_map=collate_fn)
    logger.info("Loading train_dataloader Successfully!")

    eval_dataset, eval_examples, eval_nrel = generate_relation_data(os.path.join(args.entity_output_dir, args.entity_predictions_dev), use_gold=args.eval_with_gold)
    eval_data =MNREDateset(eval_examples,tokenizer,label2id,"dev",args)
    dev_ms_dataset = ds.GeneratorDataset(source=eval_data,column_names=column_names,shuffle= False)
    dev_ms_dataset = dev_ms_dataset.batch(batch_size = args.train_batch_size,per_batch_map=collate_fn)
    logger.info("Loading eval_dataloader Successfully!")

    test_dataset, test_examples, test_nrel = generate_relation_data(os.path.join(args.entity_output_dir, args.entity_predictions_test), use_gold=args.eval_with_gold)
    test_data =MNREDateset(test_examples,tokenizer,label2id,"test",args)
    test_ms_dataset = ds.GeneratorDataset(source=test_data,column_names=column_names,shuffle= False)
    test_ms_dataset = test_ms_dataset.batch(batch_size = args.train_batch_size,per_batch_map=collate_fn)
    logger.info("Loading test_dataloader Successfully!")

    print("*"*50)
    print(args.entity_predictions_dev)
    print(os.path.join(args.entity_output_dir, args.entity_predictions_dev))
    print(os.path.join(args.entity_output_dir, args.entity_predictions_test))

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(sys.argv)
    logger.info(args)

    if os.path.exists(os.path.join(args.output_dir, 'special_tokens.json')):
        with open(os.path.join(args.output_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}

    with open(os.path.join(args.output_dir, 'special_tokens.json'), 'w') as f:
        json.dump(special_tokens, f)

    if args.do_train:

        num_train_optimization_steps = train_ms_dataset.get_dataset_size() * args.num_epoch 
        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps  = %d", num_train_optimization_steps)

        net = BertForRelation(args,num_labels)
        print("net 初始化完成")

        train_step_size = train_ms_dataset.get_dataset_size()
        lr = nn.warmup_lr(learning_rate=args.learning_rate,total_step=train_step_size * args.num_epoch,step_per_epoch=train_step_size,warmup_epoch=2)
        # args.learning_rate
        optimizer = nn.AdamWeightDecay(params=net.trainable_params(), learning_rate=lr,weight_decay=0.00)
        model = Model(net, optimizer=optimizer, eval_network=net,eval_indexes =[0,1,2],metrics={"PRF":PRF()})
        hist = {'loss':[], 'p':[], 'r':[], 'f':[]} # 训练过程记录

        train_cb=Traincallback(hist["loss"])
        eval_cb= Evalcallback(args,model,hist['p'],hist['r'],hist['f'],dev_ms_dataset,test_ms_dataset)
        model.train(args.num_epoch,train_ms_dataset,callbacks=[train_cb,eval_cb,LossMonitor(500),TimeMonitor()])
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_context_length", default=60, type=int)
    parser.add_argument("--negative_label", default="None", type=str)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_file", default=None, type=str, help="The path of the training data.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    parser.add_argument("--eval_with_gold", action="store_true", help="Whether to evaluate the relation model with gold entities provided.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=None, type=float,help="The initial learning rate for Adam.")
    parser.add_argument("--num_epoch", default=1, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--entity_output_dir", type=str, default=None, help="The directory of the prediction files of the entity model")
    parser.add_argument("--entity_predictions_dev", type=str, default="ent_pred_dev.json", help="The entity prediction file of the dev set")
    parser.add_argument("--entity_predictions_test", type=str, default="ent_pred_test.json", help="The entity prediction file of the test set")
    parser.add_argument("--prediction_file", type=str, default="predictions.json", help="The prediction filename for the relation model")
    parser.add_argument('--task', type=str, default= 'mnre')
    parser.add_argument('--question', action='store_true')
    parser.add_argument('--multiquestion', action='store_true')
    parser.add_argument('--directional', action='store_true')

    args = parser.parse_args()
    main(args)
