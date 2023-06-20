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
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader 

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')

# seed=2022
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

# torch.backends.cudnn.enabled = False 
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic=True

# os.environ['PYTHONHASHSEED'] = str(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
def output_ner_predictions(model, loader, dataset, output_file):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    span_hidden_table = {}
    tot_pred_ett = 0
    model.eval()
    with torch.no_grad():
        for i,batch in enumerate(tqdm( loader,desc=mode)):
            main_tokens_tensor,aux_tokens_tensor,main_attention_mask,aux_attention_mask,bert_spans_tensor,  spans_ner_label_tensor,attention_with_image,spans_mask_tensor,pixel_values,aux_values,rcnn_values,samples=batch
            ner_logits, spans_embedding, last_hidden = model(
                main_input_ids = main_tokens_tensor.to(device),
                spans = bert_spans_tensor.to(device),
                spans_mask = spans_mask_tensor.to(device),
                spans_ner_label = None,
                main_attention_mask = main_attention_mask.to(device),
                aux_input_ids = aux_tokens_tensor.to(device),
                aux_attention_mask = aux_attention_mask.to(device),
                attention_with_image =attention_with_image.to(device),
                pixel_values=pixel_values.to(device),
                aux_values= aux_values.to(device),
                rcnn_values= rcnn_values.to(device)
            )
            _, predicted_label = ner_logits.max(2)
            predicted_label = predicted_label.cpu().numpy()
        
            predicted = []
            for i, sample in enumerate(samples):
                ner = []
                for j in range(len(sample['spans'])):
                    ner.append(predicted_label[i][j])
                predicted.append(ner)
                
            pred_ner = predicted
            samples=batch[-1]
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

    logger.info('Total pred entities: %d'%tot_pred_ett)

    js = dataset.js
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

    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc, cls=NpEncoder) for doc in js))

def evaluate(model, loader, tot_gold,mode):
    """
    Evaluate the entity model
    """
    logger.info('Evaluating...')
    c_time = time.time()
    cor = 0
    tot_pred = 0
    l_cor = 0
    l_tot = 0
    model.eval()
    with torch.no_grad():
        for i,batch in enumerate(tqdm( loader,desc=mode)):
            main_tokens_tensor,aux_tokens_tensor,main_attention_mask,aux_attention_mask,bert_spans_tensor,  spans_ner_label_tensor,attention_with_image,spans_mask_tensor,pixel_values,aux_values,rcnn_values,samples=batch
            ner_logits, spans_embedding, last_hidden = model(
                main_input_ids = main_tokens_tensor.to(device),
                spans = bert_spans_tensor.to(device),
                spans_mask = spans_mask_tensor.to(device),
                spans_ner_label = None,
                main_attention_mask = main_attention_mask.to(device),
                aux_input_ids = aux_tokens_tensor.to(device),
                aux_attention_mask = aux_attention_mask.to(device),
                attention_with_image =attention_with_image.to(device),
                pixel_values=pixel_values.to(device),
                aux_values= aux_values.to(device),
                rcnn_values= rcnn_values.to(device)
            )
            _, predicted_label = ner_logits.max(2)
            predicted_label = predicted_label.cpu().numpy()
        
            predicted = []
            for i, sample in enumerate(samples):
                ner = []
                for j in range(len(sample['spans'])):
                    ner.append(predicted_label[i][j])
                predicted.append(ner)

            pred_ner = predicted
            samples=batch[-1]
            for sample, preds in zip(samples, pred_ner):
                for gold, pred in zip(sample['spans_label'], preds):
                    l_tot += 1
                    if pred == gold:
                        l_cor += 1
                    if pred != 0 and gold != 0 and pred == gold:
                        cor += 1
                    if pred != 0:
                        tot_pred += 1
                   
    acc = l_cor / l_tot
    logger.info('Accuracy: %5f'%acc)
    logger.info('Cor: %d, Pred TOT: %d, Gold TOT: %d'%(cor, tot_pred, tot_gold))
    p = cor / tot_pred if cor > 0 else 0.0
    r = cor / tot_gold if cor > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    logger.info('P: %.5f, R: %.5f, F1: %.5f'%(p, r, f1))
    logger.info('Used time: %f'%(time.time()-c_time))
    return f1

def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='mnre')
    parser.add_argument('--data_dir', type=str, default=None, required=True,help="path to the preprocessed dataset")
    parser.add_argument('--output_dir', type=str, default='entity_output', help="output directory of the entity model")
    parser.add_argument('--max_span_length', type=int, default=8,  help="spans w/ length up to max_span_length are considered as candidates")
    parser.add_argument('--max_context_length', type=int, default=60)

    parser.add_argument('--train_batch_size', type=int, default=32, help="batch size during training")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="learning rate for the BERT encoder")
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help="the ratio of the warmup steps to the total steps")
    parser.add_argument('--num_epoch', type=int, default=30, help="number of the training epochs")

    parser.add_argument('--print_loss_step', type=int, default=100000, help="how often logging the loss value during training")
    parser.add_argument('--eval_per_epoch', type=int, default=1, help="how often evaluating the trained model on dev set during training")

    parser.add_argument('--do_train', action='store_true',  help="whether to run training")
    parser.add_argument('--do_eval', action='store_true',  help="whether to run evaluation")
    parser.add_argument('--eval_test', action='store_true', help="whether to evaluate on test set")

    parser.add_argument('--dev_pred_filename', type=str, default="ent_pred_dev.json", help="the prediction filename for the dev set")
    parser.add_argument('--test_pred_filename', type=str, default="ent_pred_test.json", help="the prediction filename for the test set")

    parser.add_argument('--model', type=str, default='bert-base-uncased', help="the base model name (a huggingface model)")

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--question', action='store_true')
    parser.add_argument("--multiquestion",action='store_true')
    parser.add_argument("--directional",action='store_true')
    parser.add_argument("--accumulation_steps",type=int, default=1)
    parser.add_argument("--eval_step",type=int, default=50)
    parser.add_argument('--fusion_attention_drop', type=float, default=0.1)
    parser.add_argument('--fusion_hidden_drop', type=float, default=0.1)
    parser.add_argument('--fusion_attention_head', type=int, default=12)
    parser.add_argument('--fusion_num_hidden_layers', type=int, default=12)
    parser.add_argument("--max_norm",type=float, default=0.25)
    parser.add_argument("--hidden_act",type=str, default="relu")  

    args = parser.parse_args()
    args.train_data = os.path.join(args.data_dir, 'train.json')
    args.dev_data = os.path.join(args.data_dir, 'dev.json')
    args.test_data = os.path.join(args.data_dir, 'test.json')

    setseed(args.seed)
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
    model = BertForEntity(args, num_ner_labels=num_ner_labels)
    model.to(device)
    print("*"*50)
    dev_data = Dataset(args.dev_data)
    dev_samples, dev_ner = convert_dataset_to_samples(dev_data, args.max_span_length, ner_label2id=ner_label2id)
    dev_dataset =  MNREDateset(dev_samples,tokenizer,"dev",args=args)
    dev_loader =DataLoader(dataset=dev_dataset,batch_size=args.train_batch_size,shuffle= False,collate_fn=collate_fn,num_workers=5)

    test_data = Dataset(args.test_data)
    test_samples, test_ner = convert_dataset_to_samples(test_data, args.max_span_length, ner_label2id=ner_label2id)
    test_dataset =  MNREDateset(test_samples,tokenizer,"test",args=args)
    test_loader =DataLoader(dataset=test_dataset,batch_size=args.train_batch_size,shuffle= False,collate_fn=collate_fn,num_workers=5)

    if args.do_train:
        train_data = Dataset(args.train_data)
        train_samples, train_ner = convert_dataset_to_samples(train_data, args.max_span_length, ner_label2id=ner_label2id)
        train_dataset =  MNREDateset(train_samples,tokenizer,"train",args=args)
        train_loader =DataLoader(dataset=train_dataset,batch_size=args.train_batch_size,shuffle= True,collate_fn=collate_fn,num_workers=5)


        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        t_total   = len(train_loader) * args.num_epoch  // args.accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*args.warmup_proportion), t_total)
        
        tr_loss = 0
        global_step = 0
        eval_step = len(train_loader) // args.eval_per_epoch  //args.accumulation_steps
        global_step=0
        best_result = -1
        for epoch in tqdm(range(args.num_epoch)):
            model.train()
            for index,batch in enumerate(tqdm( train_loader)):
                main_tokens_tensor,aux_tokens_tensor,main_attention_mask,aux_attention_mask,bert_spans_tensor,  spans_ner_label_tensor,attention_with_image,spans_mask_tensor,pixel_values,aux_values,rcnn_values,samples=batch
                loss, __, __ = model(
                    main_input_ids = main_tokens_tensor.to(device),
                    spans = bert_spans_tensor.to(device),
                    spans_mask = spans_mask_tensor.to(device),
                    spans_ner_label = spans_ner_label_tensor.to(device),
                    main_attention_mask = main_attention_mask.to(device),
                    aux_input_ids = aux_tokens_tensor.to(device),
                    aux_attention_mask = aux_attention_mask.to(device),
                    attention_with_image =attention_with_image.to(device),
                    pixel_values=pixel_values.to(device),
                    aux_values= aux_values.to(device),
                    rcnn_values= rcnn_values.to(device)
                )
                loss = loss / args.accumulation_steps  
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_norm)
                tr_loss += loss.item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step+=1
                # if global_step % args.eval_step ==0:
            
            f1 = evaluate(model, dev_loader, dev_ner,"dev")
            if f1 > best_result:
                best_result = f1
                logger.info("!!! Best valid (epoch=%d): %.2f"%(epoch, f1*100))
                torch.save(model.state_dict(),os.path.join(args.output_dir,"best_ner.pth"))

            
    if args.do_eval:
        path=os.path.join(args.output_dir,"best_ner.pth")
        model.load_state_dict(torch.load(path))

        print("evaluate and output dev dataset result to file.")
        mode="dev"
        evaluate(model, dev_loader, dev_ner,mode)

        prediction_file = os.path.join(args.output_dir, args.dev_pred_filename)
        print("output to file")
        output_ner_predictions(model, dev_loader, dev_data, output_file=prediction_file)
        
        print("evaluate and output test dataset result to file.")
        mode="test"
        evaluate(model, test_loader, test_ner,mode)
        prediction_file = os.path.join(args.output_dir, args.test_pred_filename)
        print("output to file")
        output_ner_predictions(model, test_loader, test_data, output_file=prediction_file)