task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    'mnre':['Entity'],
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
    'mnre':["/per/per/parent","/per/per/siblings","/per/per/couple","/per/per/neighbor","/per/per/peer","/per/per/charges","/per/per/alumi",
            "/per/per/alternate_names","/per/org/member_of","/per/loc/place_of_residence","/per/loc/place_of_birth","/org/org/alternate_names","/org/org/subsidiary",
            "/org/loc/locate_at","/loc/loc/contain","/per/misc/present_in","/per/misc/awarded","/per/misc/race","/per/misc/religion","/per/misc/nationality","/misc/misc/part_of","/misc/loc/held_on"
            ]
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
