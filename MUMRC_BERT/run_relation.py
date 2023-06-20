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
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import CrossEntropyLoss

from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from relation.models import BertForRelation
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from relation.utils import generate_relation_data, decode_sample_id
from shared.const import task_rel_labels, task_ner_labels
from relation.dataloader import MNREDateset,collate_fn

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# torch.set_num_threads(6)
# seed=2022
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

# # torch.set_deterministic(True)
# torch.backends.cudnn.enabled = False 
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic=True
# # if torch.cuda.is_available():

# os.environ['PYTHONHASHSEED'] = str(seed)
def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_f1(preds, labels, e2e_ngold):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0

        if e2e_ngold is not None:
            e2e_recall = n_correct * 1.0 / e2e_ngold
            e2e_f1 = 2.0 * prec * e2e_recall / (prec + e2e_recall)
        else:
            e2e_recall = e2e_f1 = 0.0
        return {'precision': prec, 'recall': e2e_recall, 'f1': e2e_f1, 'task_recall': recall, 'task_f1': f1, 
        'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': e2e_ngold, 'task_ngold': n_gold}

def evaluate(model, device, eval_dataloader, num_labels, e2e_ngold=None, verbose=True,mode='dev'):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    cnt=0
    total_preds  = []
    total_labels = []
    for index,batch in enumerate(tqdm(eval_dataloader,desc=mode)):
        samples=batch[-1]
        batch = [t.to(device) for t in batch[:-1]]
        main_input_ids,main_attention_mask,aux_input_ids,aux_attention_mask,attention_with_image,sub_idx,obj_idx,label_ids,pixel_values,aux_values,rcnn_values=batch

        cnt=cnt+1
        with torch.no_grad():
            logits = model(main_input_ids, main_attention_mask,aux_input_ids,aux_attention_mask,attention_with_image,
                         None, sub_idx, obj_idx,pixel_values,aux_values,rcnn_values,mode=mode)
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        _,preds= logits.max(-1)
        total_preds.extend(preds.tolist())
        total_labels.extend(label_ids.tolist())

    eval_loss = eval_loss / nb_eval_steps
    result = compute_f1(total_preds,total_labels, e2e_ngold=e2e_ngold)
    result['accuracy'] = simple_accuracy(np.array(total_preds),np.array( total_labels))
    result['eval_loss'] = eval_loss
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return total_preds, result

def print_pred_json(eval_data, eval_examples, preds, id2label, output_file):
    rels = dict()
    for ex, pred in zip(eval_examples, preds):
        doc_sent, sub, obj = decode_sample_id(ex['id'])
        if doc_sent not in rels:
            rels[doc_sent] = []
        if pred != 0:
            rels[doc_sent].append([sub[0], sub[1], obj[0], obj[1], id2label[pred]])

    js = eval_data.js
    for doc in js:
        doc['predicted_relations'] = []
        for sid in range(len(doc['sentences'])):
            k = '%s@%d'%(doc['doc_key'], sid)
            doc['predicted_relations'].append(rels.get(k, []))
    
    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc) for doc in js))

def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True
    # os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(args):
    setseed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    n_gpu = torch.cuda.device_count()
    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=True)
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
    # train set
    train_dataset, train_examples, train_nrel = generate_relation_data(args.train_file, use_gold=True)
    train_data =MNREDateset(train_examples,tokenizer,label2id,"train",args)
    train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size,shuffle=True,num_workers=4,collate_fn=collate_fn)  
    logger.info("Loading train_dataloader Successfully!")

    eval_dataset, eval_examples, eval_nrel = generate_relation_data(os.path.join(args.entity_output_dir, args.entity_predictions_dev), use_gold=args.eval_with_gold)
    eval_data =MNREDateset(eval_examples,tokenizer,label2id,"dev",args)
    eval_dataloader = DataLoader(eval_data, batch_size=args.train_batch_size,shuffle=False,num_workers=4,collate_fn=collate_fn)
    logger.info("Loading eval_dataloader Successfully!")

    test_dataset, test_examples, test_nrel = generate_relation_data(os.path.join(args.entity_output_dir, args.entity_predictions_test), use_gold=args.eval_with_gold)
    test_data =MNREDateset(test_examples,tokenizer,label2id,"test",args)
    test_dataloader = DataLoader(test_data, batch_size=args.train_batch_size,shuffle=False,num_workers=4,collate_fn=collate_fn)
    logger.info("Loading test_dataloader Successfully!")

    print("*"*50)
    print(args.entity_predictions_dev)
    print(os.path.join(args.entity_output_dir, args.entity_predictions_dev))
    print(os.path.join(args.entity_output_dir, args.entity_predictions_test))
    # setseed(args.seed)


    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(sys.argv)
    logger.info(args)
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))


    if os.path.exists(os.path.join(args.output_dir, 'special_tokens.json')):
        with open(os.path.join(args.output_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}

    with open(os.path.join(args.output_dir, 'special_tokens.json'), 'w') as f:
        json.dump(special_tokens, f)

    if args.do_train:

        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs  //args.accumulation_steps
        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps  = %d", num_train_optimization_steps)


        model = BertForRelation(args,num_rel_labels=num_labels)

        model.to(device)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps * args.warmup_proportion), num_train_optimization_steps)

        global_step    = 0
        best_result = None

        for epoch in range(int(args.num_train_epochs)):
            model.train()
            
            logger.info("Start epoch #{} (lr = {})...".format(epoch, args.learning_rate))
            for step,batch in enumerate(tqdm(train_dataloader)):
                samples=batch[-1]
                batch = [t.to(device) for t in batch[:-1]]
                main_input_ids,main_attention_mask,aux_input_ids,aux_attention_mask,attention_with_image,sub_idx,obj_idx,label_ids,pixel_values,aux_values,rcnn_values=batch

                loss = model(main_input_ids, main_attention_mask, aux_input_ids,aux_attention_mask,attention_with_image,
                            label_ids, sub_idx, obj_idx,
                            pixel_values,aux_values,rcnn_values,mode='train')

                loss.backward()

                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            preds, result = evaluate(model, device, eval_dataloader, num_labels, e2e_ngold=eval_nrel,mode='dev')
            result['global_step'] = global_step
            result['epoch'] = epoch
            result['learning_rate'] = args.learning_rate
            result['batch_size'] = args.train_batch_size

            if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                best_result = result
                logger.info("!!! Best dev %s (global_step=%d): %.2f" %
                            (args.eval_metric, global_step, result[args.eval_metric] * 100.0))
                torch.save(model.state_dict(),os.path.join(args.output_dir,"re_best.pth"))
       
    evaluation_results = {}
    if args.do_eval:
        model=BertForRelation(args,num_rel_labels=num_labels)
        model.load_state_dict(torch.load(os.path.join(args.output_dir,"re_best.pth")))
        model.to(device)

        logger.info(special_tokens)

        mode='test'
        preds, result = evaluate(model, device, test_dataloader, num_labels, e2e_ngold=test_nrel,mode="test")

        logger.info('*** Evaluation Results ***')
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        logger.info("print_pred_json")
        print_pred_json(test_dataset, test_examples, preds, id2label, os.path.join(args.output_dir, args.prediction_file))

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
    parser.add_argument("--eval_metric", default="f1", type=str)
    parser.add_argument("--learning_rate", default=None, type=float,help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10%% of training.")

    parser.add_argument('--seed', type=int, default=0,help="random seed for initialization")

    parser.add_argument("--entity_output_dir", type=str, default=None, help="The directory of the prediction files of the entity model")
    parser.add_argument("--entity_predictions_dev", type=str, default="ent_pred_dev.json", help="The entity prediction file of the dev set")
    parser.add_argument("--entity_predictions_test", type=str, default="ent_pred_test.json", help="The entity prediction file of the test set")

    parser.add_argument("--prediction_file", type=str, default="predictions.json", help="The prediction filename for the relation model")

    parser.add_argument('--task', type=str, default= 'mnre')

    parser.add_argument('--question', action='store_true')
    parser.add_argument('--multiquestion', action='store_true')
    parser.add_argument('--directional', action='store_true')
    parser.add_argument("--accumulation_steps",type=int, default=1)
    parser.add_argument("--eval_step",type=int, default=50)
    parser.add_argument('--fusion_attention_drop', type=float, default=0.1)
    parser.add_argument('--fusion_hidden_drop', type=float, default=0.1)
    parser.add_argument('--fusion_attention_head', type=int, default=12)
    parser.add_argument('--fusion_num_hidden_layers', type=int, default=12)
    parser.add_argument("--max_norm",type=float, default=0.25)  
    parser.add_argument("--hidden_act",type=str, default="relu")  

    args = parser.parse_args()
    main(args)
