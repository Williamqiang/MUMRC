import json
import logging
import sys
import functools
import random
import os

from shared.data_structures import Dataset

logger = logging.getLogger('root')

def decode_sample_id(sample_id):
    doc_sent = sample_id.split('::')[0]
    pair = sample_id.split('::')[1]
    pair = pair.split('-')
    sub = (int(pair[0][1:-1].split(',')[0]), int(pair[0][1:-1].split(',')[1]))
    obj = (int(pair[1][1:-1].split(',')[0]), int(pair[1][1:-1].split(',')[1]))

    return doc_sent, sub, obj

def generate_relation_data(entity_data, use_gold=False, context_window=0):
    """
    Prepare data for the relation model
    If training: set use_gold = True
    """
    print("加载路径{}".format(entity_data))
    logger.info('Generate relation data from %s'%(entity_data))
    data = Dataset(entity_data)

    nner, nrel = 0, 0
    max_sentsample = 0
    samples = []
    for doc in data:
        for i, sent in enumerate(doc):
            sent_samples = []

            nner += len(sent.ner)
            # nrel += len(sent.relations)
            for relation in sent.relations:
                if relation.label !="None":
                    nrel+=1
                    
            if use_gold:
                sent_ner = sent.ner
            else:
                sent_ner = sent.predicted_ner
            
            gold_ner = {}
            for ner in sent.ner:
                gold_ner[ner.span] = ner.label
            
            gold_rel = {}
            for rel in sent.relations:
                gold_rel[rel.pair] = rel.label
            
            sent_start = 0
            sent_end = len(sent.text)
            tokens = sent.text
            
            for x in range(len(sent_ner)):
                for y in range(len(sent_ner)):
                    if x == y:
                        continue
                    sub = sent_ner[x]
                    obj = sent_ner[y]
                    label = gold_rel.get((sub.span, obj.span), 'None')
                    sample = {}
                    sample['docid'] = doc._doc_key
                    sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
                    sample['relation'] = label
                    sample['subj_start'] = sub.span.start_sent + sent_start
                    sample['subj_end'] = sub.span.end_sent + sent_start
                    sample['subj_type'] = sub.label
                    sample['obj_start'] = obj.span.start_sent + sent_start
                    sample['obj_end'] = obj.span.end_sent + sent_start
                    sample['obj_type'] = obj.label
                    sample['token'] = tokens
                    sample['sent_start'] = sent_start
                    sample['sent_end'] = sent_end

                    sample['aux_idx']=int(doc._doc_key)
                    sample['img_id']=doc.img_id

                    sent_samples.append(sample)

            max_sentsample = max(max_sentsample, len(sent_samples))
            samples += sent_samples
    
    tot = len(samples)
    logger.info('#samples: %d, max #sent.samples: %d'%(tot, max_sentsample))

    return data, samples, nrel
