import json
import random
import sys
import numpy as np
import torch
import unidecode
from tqdm import tqdm

__all__ = [
    "Dataloader",
]

ENTITY_PAIR_TYPE_SET = set(
    [("Chemical", "Disease"), ("Chemical", "Gene"), ("Gene", "Disease")])

# position index mapping from character level offset to token offset


def map_index(text, text_tokenized, unk_token):
    # print(text)
    # print(text_tokenized)
    sys.stdout.flush()
    ind_map = {}
    i, k = 0, 0  # (character i to token k)
    len_text = len(text)
    num_token = len(text_tokenized)
    while k < num_token:
        if i < len_text and text[i].strip() == "":
            ind_map[i] = k
            i += 1
            continue
        token = text_tokenized[k]
        if token[:2] == "##":
            token = token[2:]
        if token[:1] == "Ġ":
            token = token[1:]

        # assume that unk is always one character in the input text.
        if token != text[i:(i+len(token))]:
            ind_map[i] = k
            i += 1
            k += 1
        else:
            for _ in range(len(token)):
                ind_map[i] = k
                i += 1
            k += 1

    return ind_map


def preprocess(data_entry, tokenizer, max_text_length, relation_map, lower=True, full_annotation=True):
    # print(data_entry)
    """convert to index array, cut long sentences, cut long document, pad short sentences, pad short document"""
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    cls_token_length = len(cls_token)

    docid = data_entry["docid"]

    if "title" in data_entry and "abstract" in data_entry:
        text = data_entry["title"] + data_entry["abstract"]
        if lower == True:
            text = text.lower()
    else:
        text = data_entry["text"]
        if lower == True:
            text = text.lower()
    #text = cls_token + " " + text + " " + sep_token
    entities_info = data_entry["entity"]
    relations_info = data_entry["relation"]
    rel_vocab_size = len(relation_map)

    # tokenizer will automatically add cls and sep at the beginning and end of each sentence
    # [CLS] --> 101, [PAD] --> 0, [SEP] --> 102, [UNK] --> 100

    text = unidecode.unidecode(text)
    text_tokenized = tokenizer.tokenize(text)[:(max_text_length-2)]

    text_tokenized = [cls_token] + text_tokenized + [sep_token]
    text_wid = tokenizer.convert_tokens_to_ids(text_tokenized)

    text = cls_token + " " + text + " " + sep_token

    padid = tokenizer.pad_token_id
    input_array = np.ones(max_text_length, dtype=np.int) * int(padid)
    input_length = len(text_wid)
    input_array[0:len(text_wid)] = text_wid
    pad_array = np.array(input_array != padid, dtype=np.long)

    ind_map = map_index(text, text_tokenized, unk_token)

    # for sp, op in sorted(list(ind_map.items()), key=lambda x:x[0]):
    #     print(text[sp], text_tokenized[op], sp, op)
    # sys.stdout.flush()
    # print("")

    # create title entity indicator vector and entity type dictionary

    #if docid in ["61b3a65fb99eca31455f", "61b3a65fb983c75b2a4e", "61b3a65fb9639f197fad"]:
    #    print(text)
    #    print(text_tokenized)
    #    print(text_wid)

    sys.stdout.flush()
    entity_indicator = {}
    entity_type = {}
    entity_id_set = set([])
    for entity in entities_info:
        # if entity mention is outside max_text_length, ignore. +6 indicates additional offset due to "[CLS] "
        entity_id_set.add(entity["id"])
        entity_type[entity["id"]] = entity["type"]
        if entity["id"] not in entity_indicator:
            entity_indicator[entity["id"]] = np.zeros(max_text_length)

        if entity["start"] + cls_token_length in ind_map:
            startid = ind_map[entity["start"] + cls_token_length]
        else:
            #startid = input_length - 1
            startid = 0

        #entity_indicator[entity["id"]][startid] = 1

        if entity["end"] + cls_token_length in ind_map:
            endid = ind_map[entity["end"] + cls_token_length]
            endid += 1
        else:
            #endid = input_length - 1
            endid = input_length

        if startid >= endid: endid = startid + 1

        entity_indicator[entity["id"]][startid:endid] = 1
        #if docid in ["61b3a65fb99eca31455f", "61b3a65fb983c75b2a4e", "61b3a65fb9639f197fad"]:
        #    print(text_tokenized[startid:endid])
        #print(text_tokenized[startid], input_array[startid], startid)
        # sys.stdout.flush()

    relations_vector = {}
    relations = {}
    # print('info', relations_info)
    for rel in relations_info:
        rel_type, e1, e2 = rel["type"], rel["subj"], rel["obj"]
        if e1 in entity_indicator and e2 in entity_indicator:
            if (e1, e2) not in relations_vector:
                relations_vector[(e1, e2)] = np.zeros(rel_vocab_size)
            if rel_type in relation_map:
                # NA should not be in the relation_map to generate all-zero vector.
                relations_vector[(e1, e2)][relation_map[rel_type]] = 1
            if (e1, e2) not in relations:
                relations[(e1, e2)] = []
            relations[(e1, e2)].append(rel_type)

    # load perturbation data
    perturb_output = []
    if "perturbation" in data_entry and len(data_entry["perturbation"]) > 0:
        for perturb_data in data_entry["perturbation"]:
            if "title" in perturb_data and "abstract" in perturb_data:
                perturb_text = perturb_data["title"] + perturb_data["abstract"]
                if lower == True:
                    perturb_text = perturb_text.lower()
            else:
                perturb_text = perturb_data["text"]
                if lower == True:
                    perturb_text = perturb_text.lower()
            #text = cls_token + " " + text + " " + sep_token
            perturb_ent_info = perturb_data["entity"]

            perturb_text_tokenized = tokenizer.tokenize(
                perturb_text)[:(max_text_length-2)]

            perturb_text_tokenized = [cls_token] + \
                perturb_text_tokenized + [sep_token]
            perturb_text_wid = tokenizer.convert_tokens_to_ids(
                perturb_text_tokenized)
            perturb_text = cls_token + " " + perturb_text + " " + sep_token

            padid = tokenizer.pad_token_id
            perturb_input_array = np.ones(max_text_length,
                                          dtype=np.int) * int(padid)
            perturb_input_length = len(perturb_text_wid)
            perturb_input_array[0:len(perturb_text_wid)] = perturb_text_wid
            perturb_pad_array = np.array(
                perturb_input_array != padid, dtype=np.long)

            perturb_ind_map = map_index(
                perturb_text, perturb_text_tokenized, unk_token)

            perturb_entity_indicator = []
            for entity in perturb_ent_info:
                # if entity mention is outside max_text_length, ignore. +6 indicates additional offset due to "[CLS] "
                perturb_entity_indicator.append(np.zeros(max_text_length))

                if entity["start"] + cls_token_length in perturb_ind_map:
                    startid = perturb_ind_map[entity["start"] +
                                              cls_token_length]
                else:
                    #startid = perturb_input_length - 1
                    startid = 0

                #entity_indicator[entity["id"]][startid] = 1

                if entity["end"] + cls_token_length in perturb_ind_map:
                    endid = perturb_ind_map[entity["end"] + cls_token_length]
                    endid += 1
                else:
                    #endid = perturb_input_length - 1
                    endid = perturb_input_length

                if startid >= endid: endid = startid + 1

                perturb_entity_indicator[-1][startid:endid] = 1

            perturb_output.append({"input": perturb_input_array, "pad": perturb_pad_array, "docid": docid,
                                   "e1_indicators": perturb_entity_indicator[0], "e2_indicators": perturb_entity_indicator[1],
                                   "input_length": perturb_input_length})

    output_data = []
    sys.stdout.flush()
    if full_annotation == True:
        # in this mode, NA relation label occurs either when it is shown in the data, or there is no label between the pair
        for e1 in list(entity_id_set):
            for e2 in list(entity_id_set):
                if (entity_type[e1], entity_type[e2]) in ENTITY_PAIR_TYPE_SET:
                    #print(e1, e2, entity_type[e1], entity_type[e2])

                    e1_indicators, e2_indicators = entity_indicator[e1], entity_indicator[e2]
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
                                        "input_length": input_length,
                                        "perturbation": [],
                                        })
    else:
        # in this mode, NA relation label occurs only when it is shown in the data

        for e1, e2 in relations_vector:

            label_vector = relations_vector[(e1, e2)]
            label_names = relations[(e1, e2)]
            e1_indicators, e2_indicators = entity_indicator[e1], entity_indicator[e2]
            output_data.append({"input": input_array, "pad": pad_array, "docid": docid,
                                "label_vector": label_vector, "label_names": label_names,
                                "e1_indicators": e1_indicators, "e2_indicators": e2_indicators,
                                "e1": e1, "e2": e2,
                                "e1_type": entity_type[e1], "e2_type": entity_type[e2],
                                "input_length": input_length,
                                "perturbation": perturb_output
                                })
            #print(e1, e2, label_names, label_vector)
            sys.stdout.flush()
            # print(output_data[-1])
            sys.stdout.flush()

    sys.stdout.flush()
    return output_data


class Dataloader(object):
    """Dataloader"""

    def __init__(self, data_path, tokenizer, batchsize=1, shuffle=True, seed=0, max_text_length=512, training=False, logger=None, lowercase=True, full_annotation=True):
        # shape of input for each batch: (batchsize, max_text_length, max_sent_length)
        self.train = []
        self.val = []
        self.test = []
        self.tokenizer = tokenizer
        self.logger = logger
        # tokenizer.batch_decode(wids["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        self.relation_map = json.loads(
            open(data_path + "/relation_map.json").read())
        self.relation_name = dict([(i, r)
                                  for r, i in self.relation_map.items()])

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
                    entity_type_pair_stat[(e1t, e2t)] = {
                        "num_pos_pairs": 0, "num_neg_pairs": 0, "num_pos_rels": 0}

                num_pos_ = d["label_vector"].sum()
                if num_pos_ == 0:
                    num_neg_pairs += 1
                    entity_type_pair_stat[(e1t, e2t)]["num_neg_pairs"] += 1
                else:
                    num_pos_rels += num_pos_
                    num_pos_pairs += 1
                    entity_type_pair_stat[(
                        e1t, e2t)]["num_pos_rels"] += num_pos_
                    entity_type_pair_stat[(e1t, e2t)]["num_pos_pairs"] += 1
                    for rel_name in relation_names:
                        per_rel_stat[rel_name] += 1

            return num_pos_rels, num_pos_pairs, num_neg_pairs, entity_type_pair_stat, per_rel_stat

        if training == True:

            #with open(data_path + "/train.json", encoding="utf-8") as f:
            with open(data_path + "/train.json") as f:
                # try:
                train_json = json.loads(f.read())

                for data in tqdm(train_json[:]):
                    processed_data = preprocess(
                        data, tokenizer, max_text_length, self.relation_map, lowercase, full_annotation)
                    # print(processed_data)
                    sys.stdout.flush()
                    self.train.extend(processed_data)
                num_pos_rels, num_pos_pairs, num_neg_pairs, entity_type_pair_stat, per_rel_stat = calculate_stat(
                    self.train)
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

                # self.label2train = {}  # for re-calculating centers
                # for data in self.train:
                #    rels = data["label_names"]
                #    for r in rels:
                #        if r == "no_relation":
                #            continue
                #        if r not in self.label2train:
                #            self.label2train[r] = []
                #        self.label2train[r].append(data)

                # except:
                #    pass
        with open(data_path + "/valid.json") as f:
            # try:
            valid_json = json.loads(f.read())
            for data in tqdm(valid_json[:]):
                processed_data = preprocess(
                    data, tokenizer, max_text_length, self.relation_map, lowercase, full_annotation)
                self.val.extend(processed_data)

            num_pos_rels, num_pos_pairs, num_neg_pairs, entity_type_pair_stat, per_rel_stat = calculate_stat(
                self.val)
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
            # except:
            #    pass
        # print(len(self.train))
        with open(data_path + "/test.json") as f:
            # try:
            test_json = json.loads(f.read())
            for data in tqdm(test_json[:]):
                processed_data = preprocess(
                    data, tokenizer, max_text_length, self.relation_map, lowercase, full_annotation)
                self.test.extend(processed_data)

            num_pos_rels, num_pos_pairs, num_neg_pairs, entity_type_pair_stat, per_rel_stat = calculate_stat(
                self.test)
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
            # except:
            #     pass

        self.max_text_length = max_text_length
        self._bz = batchsize
        self._datasize = len(self.train)
        self._idx = 0
        self.num_trained_data = 0
        random.seed(seed)
        random.shuffle(self.train)
        # print(len(self.train))
        # sys.stdout.flush()

    def __len__(self):
        return self._datasize

    def __iter__(self):
        while True:
            if self._idx + self._bz > self._datasize:
                random.shuffle(self.train)
                self._idx = 0
            batch = self.train[self._idx:(self._idx+self._bz)]
            input_array, pad_array, label_array, ep_mask, e1_indicator, e2_indicator = [
            ], [], [], [], [], []
            perturb_input_array, perturb_pad_array, perturb_e1_indicator, perturb_e2_indicator = [
            ], [], [], []
            input_lengths = []
            perturb_input_lengths = []
            docids, perturb_docids = [], []
            for b in batch:
                docids.append(b["docid"])
                input_lengths.append(b["input_length"])
                input_array.append(b["input"])
                pad_array.append(b["pad"])
                label_array.append(b["label_vector"])

                e1_indicator.append(b["e1_indicators"])
                e2_indicator.append(b["e2_indicators"])
                # (text_length, text_length)
                ep_mask_ = np.full(
                    (self.max_text_length, self.max_text_length), -1e20)
                ep_outer = 1 - np.outer(b["e1_indicators"], b["e2_indicators"])
                ep_mask_ = ep_mask_ * ep_outer
                # print(b["e1_indicators"])
                # print(b["e2_indicators"])
                sys.stdout.flush()

                # ep_indices_ = [(ei, ej)
                #             for ei in np.where(b["e1_indicators"] == 1)[0]
                #             for ej in np.where(b["e2_indicators"] == 1)[0]]
                # sys.stdout.flush()
                # r, c = zip(*ep_indices_)
                # ep_mask_[r, c] = 0.0
                ep_mask.append(ep_mask_)

                if len(b["perturbation"]) > 0:
                    perturb_d = random.choice(b["perturbation"])
                    perturb_input_lengths.append(perturb_d["input_length"])
                    perturb_input_array.append(perturb_d["input"])
                    perturb_pad_array.append(perturb_d["pad"])
                    perturb_e1_indicator.append(perturb_d["e1_indicators"])
                    perturb_e2_indicator.append(perturb_d["e2_indicators"])
                    perturb_docids.append(perturb_d["docid"])
                else:
                    perturb_input_lengths.append(b["input_length"])
                    perturb_input_array.append(b["input"])
                    perturb_pad_array.append(b["pad"])
                    perturb_e1_indicator.append(b["e1_indicators"])
                    perturb_e2_indicator.append(b["e2_indicators"])

            #print(input_array[0], np.where(ep_mask[0] == 0.0), label_array[0])
            # sys.stdout.flush()
            max_length = int(np.max(input_lengths))
            input_ids = torch.tensor(np.array(input_array)[
                                     :, :max_length], dtype=torch.long)
            token_type_ids = torch.zeros_like(
                input_ids[:, :max_length], dtype=torch.long)
            attention_mask = torch.tensor(
                np.array(pad_array)[:, :max_length], dtype=torch.long)
            label_array = torch.tensor(
                np.array(label_array), dtype=torch.float)
            ep_mask = torch.tensor(
                np.array(ep_mask)[:, :max_length, :max_length], dtype=torch.float)
            e1_indicator = torch.tensor(np.array(e1_indicator)[
                                        :, :max_length], dtype=torch.float)
            e2_indicator = torch.tensor(np.array(e2_indicator)[
                                        :, :max_length], dtype=torch.float)

            perturb_max_length = int(np.max(perturb_input_lengths))
            perturb_input_ids = torch.tensor(np.array(perturb_input_array)[
                :, :perturb_max_length], dtype=torch.long)
            perturb_token_type_ids = torch.zeros_like(
                perturb_input_ids[:, :perturb_max_length], dtype=torch.long)
            perturb_attention_mask = torch.tensor(
                np.array(perturb_pad_array)[:, :perturb_max_length], dtype=torch.long)
            perturb_e1_indicator = torch.tensor(np.array(perturb_e1_indicator)[
                :, :perturb_max_length], dtype=torch.float)
            perturb_e2_indicator = torch.tensor(np.array(perturb_e2_indicator)[
                :, :perturb_max_length], dtype=torch.float)

            self._idx += self._bz
            self.num_trained_data += self._bz

            return_data = (input_ids, token_type_ids, attention_mask, ep_mask, e1_indicator, e2_indicator, label_array, docids,
                           perturb_input_ids, perturb_token_type_ids, perturb_attention_mask, perturb_e1_indicator, perturb_e2_indicator, perturb_docids)
            yield self.num_trained_data, return_data
