import json
import random
import sys
import numpy as np
import torch
from tqdm import tqdm

__all__ = [
    "Dataloader",
]

ENTITY_PAIR_TYPE_SET = set([("Chemical", "Disease"), ("Chemical", "Gene"), ("Gene", "Disease")])

def preprocess(data_entry, tokenizer, max_text_length, relation_map):
    """convert to index array, cut long sentences, cut long document, pad short sentences, pad short document"""
    docid = data_entry["docid"]
    title = data_entry["title"].lower()
    abstract = data_entry["abstract"].lower()
    entities_info = data_entry["entity"]
    relations_info = data_entry["relation"]
    rel_vocab_size = len(relation_map)


    # tokenizer will automatically add cls and sep at the beginning and end of each sentence
    # [CLS] --> 101, [PAD] --> 0, [SEP] --> 102, [UNK] --> 100

    
    text = "[CLS] " + title + abstract + " [SEP]"
    text_tokenized =tokenizer.tokenize(text)[:max_text_length]
    text_wid = tokenizer.convert_tokens_to_ids(text_tokenized)
    padid = tokenizer.vocab['[PAD]']
    input_array = np.ones(max_text_length, dtype=np.int) * int(padid)
    input_array[0:len(text_wid)] = text_wid
    pad_array = np.array(input_array != padid, dtype=np.long)

    # position index mapping from character level offset to token offset
    def map_index(text, text_tokenized):
        #print(text)
        #print(text_tokenized)
        sys.stdout.flush()
        ind_map = {}
        i, j = 0, 0
        src_str, tgt_str = "", ""
        num_token = len(text_tokenized)
        while j < num_token:
            token = text_tokenized[j]
            if token[:2] == "##": token = token[2:]
            if token != text[i:(i+len(token))]: # deal with [UNK] and space. characters of [UNK] will point to next token.
                ind_map[i] = j
                if token == "[UNK]":
                    i += 1
                    j += 1
                else:
                    i += 1
            else:
                tgt_str += token
                #print("tgt_str", tgt_str)
                while len(src_str) < len(tgt_str):
                    ind_map[i] = j
                    src_str += text[i]
                    #print("src_str", src_str)
                    sys.stdout.flush()
                    i += 1
                j += 1
        return ind_map

    ind_map = map_index(text, text_tokenized)

    # for sp, op in sorted(list(ind_map.items()), key=lambda x:x[0]):
    #     print(text[sp], text_tokenized[op], sp, op)
    # sys.stdout.flush()
    # print("")

    # create entity indicator vector and entity type dictionary
    entity_indicator = {}
    entity_type = {}
    entity_id_set = set([])
    max_length = len(text_tokenized)
    for entity in entities_info:
        # if entity mention is outside max_text_length, ignore. +6 indicates additional offset due to "[CLS] " 
        if entity["start"] + 6 in ind_map and entity["end"] + 6 in ind_map:
            entity_id_set.add(entity["id"])
            if entity["id"] not in entity_indicator:
                entity_indicator[entity["id"]] = np.zeros(max_text_length)
            #print(entity["id"], entity["start"] + 6, entity["end"] + 6, ind_map[entity["start"] + 6], ind_map[entity["end"] + 6])
            sys.stdout.flush()
            startid, endid = ind_map[entity["start"] + 6], ind_map[entity["end"] + 6]
            if startid == endid: endid += 1
            entity_indicator[entity["id"]][startid:endid] = 1
            entity_type[entity["id"]] = entity["type"]

    relations_vector = {}
    relations = {}
    for rel in relations_info:
        
        rel_type, e1, e2 = rel["type"], rel["subj"], rel["obj"]
        if e1 in entity_indicator and e2 in entity_indicator:
            if (e1, e2) not in relations_vector: relations_vector[(e1, e2)] = np.zeros(rel_vocab_size)

            relations_vector[(e1, e2)][relation_map[rel_type]] = 1

            if (e1, e2) not in relations: relations[(e1, e2)] = []
            relations[(e1, e2)].append(rel_type)

    #print(entity_id_set)
    output_data = []
    sys.stdout.flush()
    for e1 in list(entity_id_set):
        for e2 in list(entity_id_set):
            if (entity_type[e1], entity_type[e2]) in ENTITY_PAIR_TYPE_SET:
                #print(e1, e2, entity_type[e1], entity_type[e2])

                e1_indicators, e2_indicators = entity_indicator[e1], entity_indicator[e2]
                # ep_mask = np.full((max_text_length, max_text_length), -1e8) # (text_length, text_length)
                # ep_indices = [(ei, ej)
                #             for ei in np.where(e1_indicators == 1)[0]
                #             for ej in np.where(e2_indicators == 1)[0]]
                #print(entity_type[e1], e1_indicators)
                #print(entity_type[e2], e2_indicators)
                #print(ep_indices)
                # sys.stdout.flush()
                # r, c = zip(*ep_indices)
                # ep_mask[r, c] = 0.0
                if (e1, e2) in relations_vector:
                    label_vector = relations_vector[(e1, e2)]
                else:
                    label_vector = np.zeros(rel_vocab_size)
                if (e1, e2) in relations:
                    label_names = relations[(e1, e2)]
                else:
                    label_names = []
                
                output_data.append({"input": input_array, "pad": pad_array, "docid": docid, 
                                    "label_vector": label_vector, "label_names": label_names,
                                    "e1_indicators": e1_indicators, "e2_indicators": e2_indicators, 
                                    "e1": e1, "e2": e2, 
                                    "e1_type": entity_type[e1], "e2_type": entity_type[e2],
                                    })
    sys.stdout.flush()
    return output_data
    
    

class Dataloader(object):
    """Dataloader"""

    def __init__(self, data_path, tokenizer, batchsize=1, shuffle=True, seed=0, max_text_length=512, training=False, logger=None):
        # shape of input for each batch: (batchsize, max_text_length, max_sent_length)
        self.train = []
        self.val = []
        self.test = []
        self.tokenizer = tokenizer
        self.logger = logger
        # tokenizer.batch_decode(wids["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        self.relation_map = json.loads(open(data_path + "/relation_map.json").read())
        self.relation_name = dict([(i, r) for r, i in self.relation_map.items()])


        def calculate_stat(data):
            num_pos_rels = 0
            num_neg_pairs = 0
            num_pos_pairs = 0
            per_rel_stat = {}
            entity_type_pair_stat = {}
            for d in data:
                relation_names = d["label_names"]
                for rel_name in relation_names:
                    if rel_name not in per_rel_stat: 
                        per_rel_stat[rel_name] = 0
                e1t, e2t = d["e1_type"], d["e2_type"]
                if (e1t, e2t) not in entity_type_pair_stat: 
                    entity_type_pair_stat[(e1t, e2t)] = {"num_pos_pairs": 0, "num_neg_pairs": 0, "num_pos_rels": 0}

                num_pos_ = d["label_vector"].sum()
                if num_pos_ == 0:
                    num_neg_pairs += 1
                    entity_type_pair_stat[(e1t, e2t)]["num_neg_pairs"] += 1
                else:
                    num_pos_rels += num_pos_
                    num_pos_pairs += 1
                    entity_type_pair_stat[(e1t, e2t)]["num_pos_rels"] += num_pos_
                    entity_type_pair_stat[(e1t, e2t)]["num_pos_pairs"] += 1
                    for rel_name in relation_names:
                        per_rel_stat[rel_name] += 1

            return num_pos_rels, num_pos_pairs, num_neg_pairs, entity_type_pair_stat, per_rel_stat
        

        if training == True:
            with open(data_path + "/train.json") as f:
                #try:
                train_json = json.loads(f.read())
                for data in tqdm(train_json[:200]):
                    processed_data = preprocess(data, tokenizer, max_text_length, self.relation_map)
                    self.train.extend(processed_data)
                
                num_pos_rels, num_pos_pairs, num_neg_pairs, entity_type_pair_stat, per_rel_stat = calculate_stat(self.train)
                self.logger.info(f"=======================================")
                self.logger.info(f"Training: # of docs = {len(train_json)}")
                self.logger.info(f"          # of pairs = {len(self.train)}")
                self.logger.info(f"          # of positive pairs = {num_pos_pairs}")
                self.logger.info(f"          # of positive labels = {num_pos_rels}")
                self.logger.info(f"          # of negative pairs = {num_neg_pairs}")
                self.logger.info(f"---------------------------------------")
                for e1t, e2t in entity_type_pair_stat:
                    self.logger.info(f"          ({e1t}, {e2t}): # of positive pairs = {entity_type_pair_stat[(e1t, e2t)]['num_pos_pairs']}")
                    self.logger.info(f"          ({e1t}, {e2t}): # of positive labels = {entity_type_pair_stat[(e1t, e2t)]['num_pos_rels']}")
                    self.logger.info(f"          ({e1t}, {e2t}): # of negative pairs = {entity_type_pair_stat[(e1t, e2t)]['num_neg_pairs']}")
                self.logger.info(f"---------------------------------------")
                for rel_name in per_rel_stat:
                    self.logger.info(f"          {rel_name}: # of labels = {per_rel_stat[rel_name]}")
                self.logger.info(f"=======================================")
                #except:
                #    pass
        with open(data_path + "/valid.json") as f:
            try:
                valid_json = json.loads(f.read())
                for data in tqdm(valid_json[:200]):
                    processed_data = preprocess(data, tokenizer, max_text_length, self.relation_map)
                    self.val.extend(processed_data)
                num_pos_rels, num_pos_pairs, num_neg_pairs, entity_type_pair_stat, per_rel_stat = calculate_stat(self.val)
                self.logger.info(f"=======================================")
                self.logger.info(f"Valid:    # of docs = {len(valid_json)}")
                self.logger.info(f"          # of pairs = {len(self.val)}")
                self.logger.info(f"          # of positive pairs = {num_pos_pairs}")
                self.logger.info(f"          # of positive labels = {num_pos_rels}")
                self.logger.info(f"          # of negative pairs = {num_neg_pairs}")
                self.logger.info(f"---------------------------------------")
                for e1t, e2t in entity_type_pair_stat:
                    self.logger.info(f"          ({e1t}, {e2t}): # of positive pairs = {entity_type_pair_stat[(e1t, e2t)]['num_pos_pairs']}")
                    self.logger.info(f"          ({e1t}, {e2t}): # of positive labels = {entity_type_pair_stat[(e1t, e2t)]['num_pos_rels']}")
                    self.logger.info(f"          ({e1t}, {e2t}): # of negative pairs = {entity_type_pair_stat[(e1t, e2t)]['num_neg_pairs']}")
                self.logger.info(f"---------------------------------------")
                for rel_name in per_rel_stat:
                    self.logger.info(f"          {rel_name}: # of labels = {per_rel_stat[rel_name]}")
                self.logger.info(f"=======================================")
            except:
                pass
        with open(data_path + "/test.json") as f:
            try:
                test_json = json.loads(f.read())
                for data in tqdm(test_json[:200]):
                    processed_data = preprocess(data, tokenizer, max_text_length, self.relation_map)
                    self.test.extend(processed_data)
                num_pos_rels, num_pos_pairs, num_neg_pairs, entity_type_pair_stat, per_rel_stat = calculate_stat(self.test)
                self.logger.info(f"=======================================")
                self.logger.info(f"Test:     # of docs = {len(test_json)}")
                self.logger.info(f"          # of pairs = {len(self.test)}")
                self.logger.info(f"          # of positive pairs = {num_pos_pairs}")
                self.logger.info(f"          # of positive labels = {num_pos_rels}")
                self.logger.info(f"          # of negative pairs = {num_neg_pairs}")
                self.logger.info(f"---------------------------------------")
                for e1t, e2t in entity_type_pair_stat:
                    self.logger.info(f"          ({e1t}, {e2t}): # of positive pairs = {entity_type_pair_stat[(e1t, e2t)]['num_pos_pairs']}")
                    self.logger.info(f"          ({e1t}, {e2t}): # of positive labels = {entity_type_pair_stat[(e1t, e2t)]['num_pos_rels']}")
                    self.logger.info(f"          ({e1t}, {e2t}): # of negative pairs = {entity_type_pair_stat[(e1t, e2t)]['num_neg_pairs']}")
                self.logger.info(f"---------------------------------------")
                for rel_name in per_rel_stat:
                    self.logger.info(f"          {rel_name}: # of labels = {per_rel_stat[rel_name]}")
                self.logger.info(f"=======================================")
            except:
                pass

        self.max_text_length = max_text_length
        self._bz = batchsize
        self._datasize = len(self.train)
        self._idx = 0
        self.num_trained_data = 0
        random.seed(seed)
        random.shuffle(self.train)
    
    def __len__(self):
        return self._datasize
    

    def __iter__(self):
        while True:
            if self._idx + self._bz > self._datasize:
                random.shuffle(self.train)
                self._idx = 0
            batch = self.train[self._idx:(self._idx+self._bz)]
            input_array, pad_array, label_array, ep_mask = [], [], [], []
            for b in batch:
                input_array.append(b["input"])
                pad_array.append(b["pad"])
                label_array.append(b["label_vector"])

                ep_mask_ = np.full((self.max_text_length, self.max_text_length), -1e8) # (text_length, text_length)
                ep_outer = 1 - np.outer(b["e1_indicators"], b["e2_indicators"])
                ep_mask_ = ep_mask_ * ep_outer

                # ep_indices_ = [(ei, ej)
                #             for ei in np.where(b["e1_indicators"] == 1)[0]
                #             for ej in np.where(b["e2_indicators"] == 1)[0]]
                # sys.stdout.flush()
                # r, c = zip(*ep_indices_)
                # ep_mask_[r, c] = 0.0
                ep_mask.append(ep_mask_)

            input_ids = torch.tensor(np.array(input_array), dtype=torch.long)
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(np.array(pad_array), dtype=torch.long)
            label_array = torch.tensor(np.array(label_array), dtype=torch.float)
            ep_mask = torch.tensor(np.array(ep_mask), dtype=torch.float)

            self._idx += self._bz
            self.num_trained_data += self._bz
            yield self.num_trained_data, (input_ids, token_type_ids, attention_mask, ep_mask, label_array)
        
